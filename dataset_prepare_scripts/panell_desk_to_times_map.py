import re

# ---- Список панелей ----
panels = [
    {
        "title": "Fireside Chat with Manny Medina — Business Models in the Agentic Age: Monetization & Disruption",
        "start": "00:00:11.140",
        "end":   "00:28:04.210"
    },
    {
        "title": "Fireside Chat with Tiffani Bova — Reinventing the CXO Role for AI",
        "start": "00:28:04.720",
        "end":   "00:56:55.209"
    },
    {
        "title": "Fireside Chat with Latané Conant — Decoding Buyer Intent with 6AI",
        "start": "00:57:07.050",
        "end":   "01:27:06.170"
    },
    {
        "title": "Executive Roundtable — AI-Native Growth: Scaling Without the Bloat",
        "start": "01:27:07.780",
        "end":   "02:29:56.490"
    },
    {
        "title": "Fireside Chat with Jay McBain — Agentic (Headless) Ecosystems",
        "start": "02:29:56.890",
        "end":   "02:57:52.860"
    },
    {
        "title": "VC Roundtable — GPUs Over People? How VCs are Funding the AI Future",
        "start": "02:57:53.100",
        "end":   "03:56:02.610"
    },
    {
        "title": "Ask-AI Demo — A Proven Approach to AI for Revenue Teams",
        "start": "03:56:04.130",
        "end":   "04:11:46.800"
    },
    {
        "title": "Vidyard Demo — Agentic Video Messaging That Actually Converts",
        "start": "04:11:47.410",
        "end":   "04:26:35.150"
    },
    {
        "title": "Common Room Demo — Agentic Community Signals Driving Pipeline",
        "start": "04:26:36.310",
        "end":   "04:41:46.260"
    },
    {
        "title": "Momentum Demo — Agent-Led Revenue Execution in Real Time",
        "start": "04:41:47.130",
        "end":   "04:58:06.660"
    },
    {
        "title": "Amoeba AI Demo — Neuro-Symbolic Agents in Action",
        "start": "04:58:07.330",
        "end":   "05:13:46.920"
    },
    {
        "title": "Fullcast Demo — Agentic Territory Planning Without the Spreadsheet Hell",
        "start": "05:13:47.180",
        "end":   "05:28:12.470"
    },
    {
        "title": "Aviso Demo — The AI Revenue Operating System",
        "start": "05:28:14.584",
        "end":   "05:59:18.460"
    },


    # Додай усі панелі!
]

def to_seconds(t):
    h, m, s = t.split(':')
    s, ms = s.split('.')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

# Додаємо start/end в секундах
for p in panels:
    p['start_sec'] = to_seconds(p['start'])
    p['end_sec'] = to_seconds(p['end'])

def get_panel_title(time_str):
    sec = to_seconds(time_str)
    for p in panels:
        if p['start_sec'] <= sec <= p['end_sec']:
            return p['title']
    return "Unknown Panel"

with open('data_base.txt', encoding='utf-8') as fin, open('data_base_panelled3.txt', 'w', encoding='utf-8') as fout:
    panel_title = None
    for line in fin:
        # Пошук часової мітки
        match = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->", line)
        if match:
            time_str = match.group(1)
            panel_title = get_panel_title(time_str)
            fout.write(f"[PANEL] {panel_title}\n")  
        fout.write(line)
