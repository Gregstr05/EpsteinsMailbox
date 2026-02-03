"""
Features
- /ping
- /feed subscribe #channel   (per-server "epstein feed" subscription)
- /feed unsubscribe
- /feed status
- /feed post_now             (admin: send a random PDF immediately)
- /pdf random                (render first 10 pages of a random PDF)
- /pdf by_name name:<text>   (find by filename match; select if ambiguous)
- /pdf reindex               (admin: rebuild PDF directory index)

What it does
- Scans one directory (PDF_DIR) containing PDFs (recursive).
- Picks PDFs randomly or by name match.
- Renders the first MAX_PAGES pages (default 10) into JPEG images.
- Sends images + metadata as a message.
- Sends a daily post to every subscribed server/channel at DAILY_UTC_TIME.

Env (.env)
  DISCORD_TOKEN=...
  PDF_DIR=./pdfs
  DB_PATH=./bot.sqlite3
  DAILY_UTC_TIME=09:00
  DAILY_MODE=global          # global | per_guild

Optional rendering env
  MAX_PAGES=10
  RENDER_DPI=150
  RENDER_JPEG_QUALITY=85
  RENDER_MAX_EDGE_PX=1800

"""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Optional

import aiosqlite
import discord
import fitz  # PyMuPDF
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# -----------------------------
# Config
# -----------------------------

TOKEN = os.environ["DISCORD_TOKEN"]
PDF_DIR = os.environ.get("PDF_DIR", "./pdfs")
DB_PATH = os.environ.get("DB_PATH", "./bot.sqlite3")

DAILY_UTC_TIME = os.environ.get("DAILY_UTC_TIME", "09:00").strip()
DAILY_MODE = os.environ.get("DAILY_MODE", "global").strip().lower()

MAX_PAGES = int(os.environ.get("MAX_PAGES", "10"))
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))
RENDER_JPEG_QUALITY = int(os.environ.get("RENDER_JPEG_QUALITY", "85"))
RENDER_MAX_EDGE_PX = int(os.environ.get("RENDER_MAX_EDGE_PX", "1800"))

# Limits CPU burn if many people spam commands
RENDER_CONCURRENCY = int(os.environ.get("RENDER_CONCURRENCY", "2"))


def parse_hhmm_utc(s: str) -> dt.time:
    hh, mm = s.split(":")
    return dt.time(
        hour=int(hh),
        minute=int(mm)
    )


DAILY_TIME_UTC = parse_hhmm_utc(DAILY_UTC_TIME)


# -----------------------------
# DB (subscriptions)
# -----------------------------


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_subscriptions (
              guild_id INTEGER PRIMARY KEY,
              channel_id INTEGER NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1,
              last_sent_ymd TEXT
            )
            """
        )
        await db.commit()


async def set_subscription(guild_id: int, channel_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO daily_subscriptions (guild_id, channel_id, enabled)
            VALUES (?, ?, 1)
            ON CONFLICT(guild_id) DO UPDATE SET
              channel_id=excluded.channel_id,
              enabled=1
            """,
            (guild_id, channel_id),
        )
        await db.commit()


async def disable_subscription(guild_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE daily_subscriptions
            SET enabled=0
            WHERE guild_id=?
            """,
            (guild_id,),
        )
        await db.commit()


async def get_subscription(
    guild_id: int,
) -> Optional[tuple[int, int, Optional[str]]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT channel_id, enabled, last_sent_ymd
            FROM daily_subscriptions
            WHERE guild_id=?
            """,
            (guild_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        channel_id, enabled, last_sent_ymd = row
        return int(channel_id), int(enabled), last_sent_ymd


async def mark_sent_today(guild_id: int, ymd: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE daily_subscriptions
            SET last_sent_ymd=?
            WHERE guild_id=?
            """,
            (ymd, guild_id),
        )
        await db.commit()


async def get_due_subscriptions(now_utc: dt.datetime) -> list[tuple[int, int]]:
    today = now_utc.date().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT guild_id, channel_id
            FROM daily_subscriptions
            WHERE enabled=1
              AND (last_sent_ymd IS NULL OR last_sent_ymd != ?)
            """,
            (today,),
        )
        rows = await cur.fetchall()
        return [(int(gid), int(cid)) for (gid, cid) in rows]


# -----------------------------
# PDF index
# -----------------------------


@dataclass(frozen=True)
class PdfEntry:
    path: Path
    rel: str
    stem_cf: str


class PdfIndex:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self._lock = asyncio.Lock()
        self.entries: list[PdfEntry] = []
        self.by_stem: DefaultDict[str, list[PdfEntry]] = defaultdict(list)

    def _scan(self) -> tuple[list[PdfEntry], DefaultDict[str, list[PdfEntry]]]:
        if not self.root.exists():
            raise FileNotFoundError(f"PDF_DIR not found: {self.root.resolve()}")

        paths = [p for p in self.root.rglob("*.pdf") if p.is_file()]
        entries: list[PdfEntry] = []
        by_stem: DefaultDict[str, list[PdfEntry]] = defaultdict(list)

        for p in paths:
            rel = p.relative_to(self.root).as_posix()
            stem_cf = p.stem.casefold()
            e = PdfEntry(path=p, rel=rel, stem_cf=stem_cf)
            entries.append(e)
            by_stem[stem_cf].append(e)

        return entries, by_stem

    async def rebuild(self) -> int:
        entries, by_stem = await asyncio.to_thread(self._scan)
        async with self._lock:
            self.entries = entries
            self.by_stem = by_stem
            return len(entries)

    async def random_entry(self) -> PdfEntry:
        async with self._lock:
            if not self.entries:
                raise RuntimeError("No PDFs indexed.")
            return random.choice(self.entries)

    async def find_entries(self, query: str) -> list[PdfEntry]:
        q = query.strip().casefold()
        if not q:
            return []

        async with self._lock:
            # Exact stem match first
            exact = list(self.by_stem.get(q, []))
            if exact:
                return exact

            # Otherwise substring on stem
            hits: list[PdfEntry] = []
            for stem, entries in self.by_stem.items():
                if q in stem:
                    hits.extend(entries)
            return hits

    async def autocomplete_stems(
        self,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        q = (current or "").strip().casefold()
        if not q:
            return []

        async with self._lock:
            stems = list(self.by_stem.keys())

        out: list[app_commands.Choice[str]] = []
        for stem in stems:
            if q in stem:
                out.append(app_commands.Choice(name=stem[:100], value=stem))
                if len(out) >= 25:
                    break
        return out


# -----------------------------
# Rendering
# -----------------------------


@dataclass
class RenderResult:
    embed: discord.Embed
    files: list[discord.File]


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(v)} {u}"
            return f"{v:.1f} {u}"
        v /= 1024
    return f"{n} B"


def _render_pdf_first_pages_to_jpegs(
    pdf_bytes: bytes,
    *,
    max_pages: int,
    dpi: int,
    jpeg_quality: int,
    max_edge_px: int,
) -> tuple[list[bytes], dict, int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    meta = doc.metadata or {}
    page_count = doc.page_count

    out: list[bytes] = []
    limit = min(page_count, max_pages)

    for i in range(limit):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi, alpha=False)

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        w, h = img.size
        scale = min(1.0, max_edge_px / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )
        out.append(buf.getvalue())

    return out, meta, page_count


def _build_embed(
    *,
    title_prefix: str,
    entry: Optional[PdfEntry],
    meta: dict,
    page_count: int,
    max_pages: int,
) -> discord.Embed:
    title = meta.get("title") or "Untitled PDF"
    author = meta.get("author") or "Unknown"

    desc = [
        f"**PDF:** {entry.rel if entry else 'n/a'}",
        #f"**Title:** {title}",
        #f"**Author:** {author}",
        f"**Pages:** {page_count} (showing first {min(page_count, max_pages)})",
    ]

    embed = discord.Embed(
        title=title_prefix,
        description="\n".join(desc),
        color=discord.Color.blurple(),
    )

    if entry:
        try:
            st = entry.path.stat()
            embed.add_field(name="File size", value=_human_bytes(st.st_size))
            embed.add_field(
                name="Modified",
                value=dt.datetime.fromtimestamp(st.st_mtime).isoformat(
                    timespec="seconds"
                ),
                inline=False,
            )
        except OSError:
            pass

    return embed


async def render_entry(
    *,
    entry: PdfEntry,
    title_prefix: str,
    semaphore: asyncio.Semaphore,
) -> RenderResult:
    pdf_bytes = entry.path.read_bytes()

    async with semaphore:
        images, meta, page_count = await asyncio.to_thread(
            _render_pdf_first_pages_to_jpegs,
            pdf_bytes,
            max_pages=MAX_PAGES,
            dpi=RENDER_DPI,
            jpeg_quality=RENDER_JPEG_QUALITY,
            max_edge_px=RENDER_MAX_EDGE_PX,
        )

    embed = _build_embed(
        title_prefix=title_prefix,
        entry=entry,
        meta=meta,
        page_count=page_count,
        max_pages=MAX_PAGES,
    )

    files: list[discord.File] = []
    for idx, img_bytes in enumerate(images, start=1):
        fp = io.BytesIO(img_bytes)
        filename = f"{Path(entry.rel).stem}_p{idx:02d}.jpg"
        files.append(discord.File(fp=fp, filename=filename))

    return RenderResult(embed=embed, files=files)


# -----------------------------
# UI for ambiguous matches
# -----------------------------


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


class PdfPickView(discord.ui.View):
    def __init__(
        self,
        *,
        author_id: int,
        entries: list[PdfEntry],
        on_pick,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(timeout=timeout)
        self.author_id = author_id
        self.entries = entries[:25]
        self.on_pick = on_pick

        options: list[discord.SelectOption] = []
        for i, e in enumerate(self.entries):
            label = _truncate(e.rel, 100)
            options.append(discord.SelectOption(label=label, value=str(i)))

        self.select = discord.ui.Select(
            placeholder="Pick the PDF you meant…",
            min_values=1,
            max_values=1,
            options=options,
        )
        self.select.callback = self._select_callback
        self.add_item(self.select)

    async def _select_callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only the command invoker can use this menu.",
                ephemeral=True,
            )
            return

        idx = int(self.select.values[0])
        entry = self.entries[idx]
        await self.on_pick(interaction, entry)
        self.stop()


# -----------------------------
# Bot
# -----------------------------


class PdfNewsBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)

        self.pdf_index = PdfIndex(PDF_DIR)
        self.render_sem = asyncio.Semaphore(RENDER_CONCURRENCY)

    async def setup_hook(self) -> None:
        await init_db()
        count = await self.pdf_index.rebuild()
        print(f"Indexed {count} PDFs from {self.pdf_index.root.resolve()}")

        await self.tree.sync()
        self.daily_loop.start()

    @tasks.loop(seconds=30)
    async def daily_loop(self) -> None:
        # now_utc is aware, but .time() returns a naive object representing UTC
        now_utc = dt.datetime.now(tz=dt.timezone.utc)
        current_time_naive = now_utc.time()

        # Compare naive current UTC time to naive target UTC time
        if current_time_naive < DAILY_TIME_UTC:
            return

        due = await get_due_subscriptions(now_utc)
        if not due:
            return

        today = now_utc.date().isoformat()

        shared_result: Optional[RenderResult] = None
        shared_entry: Optional[PdfEntry] = None

        if DAILY_MODE == "global":
            shared_entry = await self.pdf_index.random_entry()
            shared_result = await render_entry(
                entry=shared_entry,
                title_prefix=f"Daily: {shared_entry.rel.removesuffix('.pdf')}",
                semaphore=self.render_sem,
            )

        for guild_id, channel_id in due:
            try:
                channel = self.get_channel(channel_id)
                if channel is None:
                    channel = await self.fetch_channel(channel_id)

                if not isinstance(channel, discord.abc.Messageable):
                    continue

                if DAILY_MODE == "per_guild":
                    entry = await self.pdf_index.random_entry()
                    result = await render_entry(
                        entry=entry,
                        title_prefix=f"Daily: {entry.rel.removesuffix('.pdf')}",
                        semaphore=self.render_sem,
                    )
                else:
                    if shared_result is None:
                        continue
                    result = shared_result

                await channel.send(embed=result.embed, files=result.files)
                await mark_sent_today(guild_id, today)
            except Exception as e:
                print(f"Daily send failed to {guild_id}/{channel_id}: {e}")

    @daily_loop.before_loop
    async def before_daily_loop(self) -> None:
        await self.wait_until_ready()


bot = PdfNewsBot()

# -----------------------------
# Commands
# -----------------------------


@bot.tree.command(name="ping", description="Simple test command.")
async def ping(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("Nice animation")


feed_group = app_commands.Group(
    name="feed",
    description="Subscribe/unsubscribe this server to the daily PDF feed.",
)


@feed_group.command(name="subscribe", description="Subscribe a channel to the feed.")
@app_commands.checks.has_permissions(manage_guild=True)
async def feed_subscribe(
    interaction: discord.Interaction,
    channel: discord.TextChannel,
) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(
            "This must be used in a server.",
            ephemeral=True,
        )
        return

    await set_subscription(interaction.guild.id, channel.id)
    await interaction.response.send_message(
        f"Subscribed: daily PDFs will be posted in {channel.mention} "
        f"at {DAILY_UTC_TIME} UTC.",
        ephemeral=True,
    )


@feed_group.command(name="unsubscribe", description="Disable the feed for this server.")
@app_commands.checks.has_permissions(manage_guild=True)
async def feed_unsubscribe(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(
            "This must be used in a server.",
            ephemeral=True,
        )
        return

    await disable_subscription(interaction.guild.id)
    await interaction.response.send_message(
        "Unsubscribed: daily PDFs are disabled for this server.",
        ephemeral=True,
    )


@feed_group.command(name="status", description="Show this server's feed status.")
async def feed_status(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(
            "This must be used in a server.",
            ephemeral=True,
        )
        return

    sub = await get_subscription(interaction.guild.id)
    if sub is None:
        await interaction.response.send_message(
            "Not subscribed.",
            ephemeral=True,
        )
        return

    channel_id, enabled, last_sent = sub
    ch = interaction.guild.get_channel(channel_id)
    ch_display = ch.mention if isinstance(ch, discord.TextChannel) else str(channel_id)

    await interaction.response.send_message(
        f"Channel: {ch_display}\n"
        f"Enabled: {'yes' if enabled else 'no'}\n"
        f"Last sent (UTC date): {last_sent or 'never'}\n"
        f"Daily time: {DAILY_UTC_TIME} UTC\n"
        f"Mode: {DAILY_MODE}",
        ephemeral=True,
    )


@feed_group.command(name="post_now", description="Post a random PDF right now.")
@app_commands.checks.has_permissions(manage_guild=True)
async def feed_post_now(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(
            "This must be used in a server.",
            ephemeral=True,
        )
        return

    sub = await get_subscription(interaction.guild.id)
    if sub is None or sub[1] == 0:
        await interaction.response.send_message(
            "This server is not subscribed. Use /feed subscribe first.",
            ephemeral=True,
        )
        return

    channel_id = sub[0]
    await interaction.response.defer(thinking=True, ephemeral=True)

    entry = await interaction.client.pdf_index.random_entry()
    result = await render_entry(
        entry=entry,
        title_prefix=f"{entry.rel.removesuffix('.pdf')}",
        semaphore=interaction.client.render_sem,
    )

    channel = interaction.guild.get_channel(channel_id)
    if channel is None:
        channel = await interaction.client.fetch_channel(channel_id)

    if not isinstance(channel, discord.abc.Messageable):
        await interaction.followup.send("Configured channel is not messageable.")
        return

    await channel.send(embed=result.embed, files=result.files)
    await interaction.followup.send(f"Posted in <#{channel_id}>.")


bot.tree.add_command(feed_group)

pdf_group = app_commands.Group(
    name="pdf",
    description="Render PDFs from the bot's PDF directory.",
)


@pdf_group.command(name="random", description="Render a random PDF (first pages).")
async def pdf_random(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)

    entry = await interaction.client.pdf_index.random_entry()
    result = await render_entry(
        entry=entry,
        title_prefix=f"Random: {entry.rel.removesuffix('.pdf')}",
        semaphore=interaction.client.render_sem,
    )
    await interaction.followup.send(embed=result.embed, files=result.files)


async def pdf_name_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    return await interaction.client.pdf_index.autocomplete_stems(current)

@pdf_group.command(
    name="by_name",
    description="Render a PDF by matching its filename (stem).",
)
@app_commands.describe(name="Filename to match (without .pdf).")
@app_commands.autocomplete(name=pdf_name_autocomplete)
async def pdf_by_name(interaction: discord.Interaction, name: str) -> None:
    matches = await interaction.client.pdf_index.find_entries(name)

    if not matches:
        await interaction.response.send_message(
            f'No PDFs found matching "{name}".',
            ephemeral=True,
        )
        return

    if len(matches) == 1:
        await interaction.response.defer(thinking=True)
        entry = matches[0]
        result = await render_entry(
            entry=entry,
            title_prefix=f"PDF: {entry.rel}",
            semaphore=interaction.client.render_sem,
        )
        await interaction.followup.send(embed=result.embed, files=result.files)
        return

    # Ambiguous: show picker then render on selection.
    async def on_pick(
        pick_interaction: discord.Interaction,
        entry: PdfEntry,
    ) -> None:
        await pick_interaction.response.defer(thinking=True)
        result = await render_entry(
            entry=entry,
            title_prefix=f"PDF: {entry.rel}",
            semaphore=pick_interaction.client.render_sem,
        )
        await pick_interaction.followup.send(embed=result.embed, files=result.files)

    view = PdfPickView(
        author_id=interaction.user.id,
        entries=matches,
        on_pick=on_pick,
    )

    await interaction.response.send_message(
        f'Found {len(matches)} matches for "{name}". Choose one:',
        view=view,
        ephemeral=True,
    )


@pdf_group.command(
    name="reindex",
    description="Rebuild the PDF directory index (admin).",
)
@app_commands.checks.has_permissions(manage_guild=True)
async def pdf_reindex(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("I legally agree")


@bot.tree.command(
    name="agree",
    description="agree",
)
async def pdf_agree(interaction: discord.Interaction):
    await interaction.response.send_message(f"I legally agree")

bot.tree.add_command(pdf_group)

if __name__ == "__main__":
    bot.run(TOKEN)