"""Spend and savings reporting (JSON + a simple HTML page)."""

from __future__ import annotations

import html
import time

from . import selfhost
from .db import Store


def summary(store: Store, days: int = 30) -> dict:
    since = time.time() - days * 86400
    q = lambda sql: store.query(sql, (since,))  # noqa: E731

    totals = q(
        """SELECT COUNT(*) AS requests,
                  COALESCE(SUM(cost_usd),0) AS cost,
                  COALESCE(SUM(baseline_cost_usd),0) AS baseline,
                  COALESCE(SUM(input_tokens),0) AS input_tokens,
                  COALESCE(SUM(output_tokens),0) AS output_tokens,
                  COALESCE(SUM(cache_creation_tokens),0) AS cache_write_tokens,
                  COALESCE(SUM(cache_read_tokens),0) AS cache_read_tokens,
                  COALESCE(SUM(trimmed_chars),0) AS trimmed_chars,
                  COALESCE(SUM(deduped_chars),0) AS deduped_chars,
                  COALESCE(SUM(cache_breakpoints_added),0) AS breakpoints_added,
                  COUNT(DISTINCT engineer) AS engineers
           FROM requests WHERE ts >= ?"""
    )[0]
    own = selfhost.report(store, since)
    if own:  # the GPU is a cost too, even though it isn't billed per request
        totals["cost"] += own["gpu_cost"]
    totals["savings"] = totals["baseline"] - totals["cost"]
    totals["savings_pct"] = (100 * totals["savings"] / totals["baseline"]) if totals["baseline"] else 0.0
    all_input = totals["input_tokens"] + totals["cache_read_tokens"] + totals["cache_write_tokens"]
    totals["cache_hit_pct"] = (100 * totals["cache_read_tokens"] / all_input) if all_input else 0.0

    by_engineer = q(
        """SELECT engineer, COALESCE(team,'') AS team, COUNT(*) AS requests,
                  SUM(cost_usd) AS cost, SUM(baseline_cost_usd) AS baseline,
                  SUM(output_tokens) AS output_tokens
           FROM requests WHERE ts >= ? GROUP BY engineer, team ORDER BY cost DESC"""
    )
    by_model = q(
        """SELECT model_used AS model, COUNT(*) AS requests, SUM(cost_usd) AS cost
           FROM requests WHERE ts >= ? GROUP BY model_used ORDER BY cost DESC"""
    )
    by_day = q(
        """SELECT date(ts,'unixepoch') AS day, COUNT(*) AS requests,
                  SUM(cost_usd) AS cost, SUM(baseline_cost_usd) AS baseline
           FROM requests WHERE ts >= ? GROUP BY day ORDER BY day"""
    )
    routing = q(
        """SELECT route_decision AS decision, COUNT(*) AS requests, SUM(cost_usd) AS cost,
                  SUM(COALESCE(cost_usd - shadow_cost_usd, 0)) AS shadow_potential_savings
           FROM requests WHERE ts >= ? GROUP BY route_decision"""
    )
    by_provider = q(
        """SELECT COALESCE(provider,'anthropic') AS provider, COUNT(*) AS requests,
                  SUM(cost_usd) AS cost, SUM(baseline_cost_usd) AS baseline
           FROM requests WHERE ts >= ? GROUP BY provider ORDER BY cost DESC"""
    )
    by_feature = q(
        """SELECT COALESCE(feature,'(unlabeled)') AS feature, COUNT(*) AS requests,
                  SUM(cost_usd) AS cost, SUM(baseline_cost_usd) AS baseline,
                  SUM(COALESCE(audio_tokens,0)) AS audio_tokens
           FROM requests WHERE ts >= ? GROUP BY feature ORDER BY cost DESC"""
    )
    cache = q(
        """SELECT COUNT(*) AS hits, COALESCE(SUM(baseline_cost_usd),0) AS saved
           FROM requests WHERE ts >= ? AND route_decision = 'cache-hit'"""
    )[0]
    for rows in (by_engineer, by_day, by_provider, by_feature):
        for r in rows:
            r["savings"] = (r.get("baseline") or 0) - (r.get("cost") or 0)

    return {
        "days": days,
        "totals": totals,
        "by_engineer": by_engineer,
        "by_model": by_model,
        "by_provider": by_provider,
        "by_feature": by_feature,
        "response_cache": cache,
        "by_day": by_day,
        "routing": routing,
        "self_hosted": own,
    }


def _money(x) -> str:
    x = x or 0
    sign, x = ("-" if x < 0 else ""), abs(x)
    return f"{sign}${x:,.4f}" if 0 < x < 0.01 else f"{sign}${x:,.2f}"  # small amounts stay visible


def _money_inr(x) -> str:
    rate = selfhost.inr_per_usd()
    return _money(x) + (f" (₹{(x or 0) * rate:,.0f})" if rate else "")


def _self_hosted_section(r: dict | None) -> str:
    if not r:
        return ""
    vs = ", ".join(r["compare_to"])
    rows = [["Requests to your own models", f"{r['requests']:,}"],
            [f"Same requests on {vs}", _money_inr(r["api_cost"])]]
    if r["gpu_cost_per_hour"]:
        rows.append(["Your GPU", f"{r['gpu_hours']:.1f} h × {_money_inr(r['gpu_cost_per_hour'])}/h = "
                                 f"{_money_inr(r['gpu_cost'])}"])
    if r["token_cost"]:
        rows.append(["Per-token price set for own models", _money_inr(r["token_cost"])])
    saved = r["saved"]
    rows.append(["Result", f"saved {_money_inr(saved)}" if saved >= 0 else
                 f"the API would have been cheaper by {_money_inr(-saved)}"])
    table = _table(["", ""], rows)
    if r["breakeven_per_hour"]:
        note = (f"The GPU costs the same per hour however busy it is. At your current request size "
                f"({_money(r['api_cost_per_request'] * 1000)} per 1,000 requests on {vs}), it pays off above "
                f"<b>{r['breakeven_per_hour']:,.0f} requests per hour</b>. Busiest hour so far: "
                f"{r['busiest_hour']:,}. Faster serving (vLLM on an L4) raises how many it can handle.")
    elif not r["gpu_cost_per_hour"]:
        note = "Set TOKENWISE_GPU_COST_PER_HOUR to include what the GPU costs."
    else:
        note = "Break-even appears after the first requests."
    return f"<h2>Own GPU vs hosted API</h2>{table}<p class=note>{note}</p>"


def _table(headers: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs or '<tr><td colspan=99>No data yet</td></tr>'}</tbody></table>"


def render_html(store: Store, days: int = 30) -> str:
    s = summary(store, days)
    t = s["totals"]
    shadow = sum(r["shadow_potential_savings"] or 0 for r in s["routing"] if r["decision"] == "shadow")
    tiles = [
        ("Spend", _money(t["cost"])),
        ("Without Tokenwise (est.)", _money(t["baseline"])),
        ("Saved", f"{_money(t['savings'])} ({t['savings_pct']:.1f}%)" if t["savings"] >= 0
                  else f"{_money(t['savings'])} (API cheaper)"),
        ("Cache hit rate", f"{t['cache_hit_pct']:.1f}%"),
        ("Requests", f"{t['requests']:,}"),
        ("Engineers", f"{t['engineers']:,}"),
    ]
    tile_html = "".join(f"<div class=tile><div class=k>{k}</div><div class=v>{v}</div></div>" for k, v in tiles)

    eng = _table(
        ["Engineer", "Team", "Requests", "Spend", "Saved", "Output tokens"],
        [[r["engineer"], r["team"], r["requests"], _money(r["cost"]), _money(r["savings"]), f"{r['output_tokens'] or 0:,}"]
         for r in s["by_engineer"]],
    )
    providers = _table(
        ["Provider", "Requests", "Spend", "Saved"],
        [[r["provider"], r["requests"], _money(r["cost"]), _money(r["savings"])] for r in s["by_provider"]],
    )
    features = _table(
        ["Feature", "Requests", "Spend", "Saved", "Audio tokens"],
        [[r["feature"], r["requests"], _money(r["cost"]), _money(r["savings"]), f"{r['audio_tokens'] or 0:,}"]
         for r in s["by_feature"]],
    )
    rc = s["response_cache"]
    cache_note = (
        f"<p class=note>Response cache: {rc['hits']:,} repeat requests answered from cache, "
        f"saving {_money(rc['saved'])}.</p>" if rc["hits"] else ""
    )
    models = _table(["Model", "Requests", "Spend"], [[r["model"], r["requests"], _money(r["cost"])] for r in s["by_model"]])
    daily = _table(
        ["Day", "Requests", "Spend", "Saved"],
        [[r["day"], r["requests"], _money(r["cost"]), _money(r["savings"])] for r in s["by_day"]],
    )
    routing = _table(
        ["Routing", "Requests", "Spend"], [[r["decision"], r["requests"], _money(r["cost"])] for r in s["routing"]]
    )
    shadow_note = (
        f"<p class=note>Shadow mode: routing would have saved about <b>{_money(shadow)}</b> "
        f"(priced on the same token counts; the cheap model's real output length will differ).</p>"
        if shadow else ""
    )
    opt_note = (
        f"<p class=note>Trimmed {t['trimmed_chars']:,} chars of tool output, removed {t['deduped_chars']:,} chars of "
        f"duplicate output, added {t['breakpoints_added']:,} cache breakpoints.</p>"
    )
    from .workflows import summary as wf_summary  # local import: workflows imports nothing from here

    wf_rows = wf_summary(store, days)
    wf = _table(
        ["Client", "Workflow", "Requests", "Success", "👍 / 👎", "Corrected", "Retried", "Trainable", "Spend",
         "Avg in / out tokens", "Top tool sequence"],
        [[r["client"], r["workflow_label"], r["requests"], f"{r['success_pct']}%", f"{r['good']} / {r['bad']}",
          r["corrected"], r["retried"], r["trainable"], _money(r["cost"]),
          f"{int(r['avg_in'] or 0):,} / {int(r['avg_out'] or 0):,}",
          (r["top_tool_sequences"][0]["sequence"] if r["top_tool_sequences"] else "")] for r in wf_rows],
    )
    wf_section = (f"<h2>Workflows (captured traces)</h2>{wf}<p class=note>Trainable = succeeded and not rated "
                  f"bad, or corrected by a person. Export with: python -m tokenwise.export</p>") if wf_rows else ""
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Tokenwise</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#1f2328;background:#fafafa}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:16px}}
.tile{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px}}
.k{{font-size:12px;color:#6b7280}} .v{{font-size:20px;font-weight:600;margin-top:4px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:14px}}
th,td{{text-align:left;padding:8px;border-bottom:1px solid #eee}} th{{color:#6b7280;font-weight:500}}
.note{{color:#6b7280;font-size:13px}}
</style></head><body>
<h1>Tokenwise</h1><div class=note>Last {days} days · savings are estimates</div>
<div class=tiles>{tile_html}</div>
{opt_note}{cache_note}
{_self_hosted_section(s["self_hosted"])}
{wf_section}
<h2>By provider</h2>{providers}
<h2>By feature</h2>{features}
<h2>By engineer</h2>{eng}
<h2>By model</h2>{models}
<h2>Routing</h2>{routing}{shadow_note}
<h2>By day</h2>{daily}
</body></html>"""
