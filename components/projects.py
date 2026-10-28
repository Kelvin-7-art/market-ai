"""Projects workspace."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from market_ai.components.ui.cards import avatar_stack, badge, stat_card
from market_ai.components.ui.feedback import empty_state, toast_success
from market_ai.components.ui.inputs import search_bar


PROJECT_TEMPLATES = ["Blank", "Stock Analysis", "Portfolio Tracker", "ML Pipeline"]


def _seed_projects() -> list[dict[str, object]]:
    return [
        {"id": "p1", "name": "US Large Cap Signal Lab", "status": "Active", "type": "Analysis", "description": "Factor signals and regime analysis across the S&P 500 universe.", "tags": ["SPX", "Factors", "ML"], "modified": "12 min ago", "files": 28, "models": 4, "collaborators": ["A", "M", "R"], "cover": "linear-gradient(135deg,#173b63,#236e82)", "icon": "↗", "starred": True, "pinned": True, "accuracy": "94.8%", "forecast": "+1.8%"},
        {"id": "p2", "name": "Crypto Momentum Monitor", "status": "Active", "type": "Portfolio", "description": "Systematic momentum signals for liquid crypto pairs with risk overlays.", "tags": ["BTC", "Signals", "Risk"], "modified": "2 hr ago", "files": 16, "models": 3, "collaborators": ["K", "A"], "cover": "linear-gradient(135deg,#3d2869,#793d8c)", "icon": "ϟ", "starred": True, "pinned": False, "accuracy": "91.2%", "forecast": "-0.4%"},
        {"id": "p3", "name": "Earnings Season 2026", "status": "Draft", "type": "Research", "description": "Pre-earnings volatility screens and post-event drift research.", "tags": ["Earnings", "Volatility"], "modified": "Yesterday", "files": 9, "models": 1, "collaborators": ["A"], "cover": "linear-gradient(135deg,#163f3a,#22705e)", "icon": "▤", "starred": False, "pinned": False, "accuracy": "88.4%", "forecast": "+0.7%"},
        {"id": "p4", "name": "Macro Regime Classifier", "status": "Active", "type": "Model Training", "description": "Classifying growth, inflation and liquidity regimes from macro series.", "tags": ["Macro", "NLP", "Research"], "modified": "3 days ago", "files": 42, "models": 6, "collaborators": ["M", "R", "J"], "cover": "linear-gradient(135deg,#312555,#4b4997)", "icon": "◇", "starred": False, "pinned": True, "accuracy": "92.6%", "forecast": "+0.9%"},
        {"id": "p5", "name": "Energy Relative Value", "status": "Archived", "type": "Backtest", "description": "Cross-sectional value and carry research across global energy markets.", "tags": ["Energy", "Pairs"], "modified": "14 days ago", "files": 21, "models": 2, "collaborators": ["A", "J"], "cover": "linear-gradient(135deg,#4b2930,#9e4c52)", "icon": "◎", "starred": False, "pinned": False, "accuracy": "86.7%", "forecast": "Archived"},
        {"id": "p6", "name": "Portfolio Stress Tests", "status": "Active", "type": "Portfolio", "description": "Scenario analysis for tail risk, liquidity and concentration exposure.", "tags": ["Portfolio", "Risk"], "modified": "18 days ago", "files": 13, "models": 3, "collaborators": ["A", "K"], "cover": "linear-gradient(135deg,#164c5b,#206e98)", "icon": "▦", "starred": False, "pinned": False, "accuracy": "90.1%", "forecast": "-2.1%"},
    ]


def _projects() -> list[dict[str, object]]:
    if "projects" not in st.session_state:
        st.session_state.projects = _seed_projects()
    return st.session_state.projects


@st.dialog("Create a new project")
def _create_project_dialog() -> None:
    """Collect project details in a native Streamlit dialog."""
    name = st.text_input("Project name", placeholder="e.g. European rates dashboard")
    description = st.text_area("Description", placeholder="What are you investigating?", height=90)
    template = st.selectbox("Start from a template", PROJECT_TEMPLATES + ["Research Notebook", "Backtest Framework"])
    visibility = st.radio("Visibility", ["Private", "Team", "Public"], horizontal=True)
    tags = st.text_input("Tags", placeholder="Rates, Macro, Research")
    icon = st.selectbox("Project icon", ["＋", "↗", "ϟ", "▤", "◇", "◎", "▦"])
    ticker = st.text_input("Default ticker", "SPY")
    invite = st.text_input("Invite collaborators", placeholder="Add email addresses, separated by commas")
    training = st.toggle("Enable model training", value=True)
    auto_sync = st.toggle("Auto-sync with Git", value=False)
    if st.button("Create project", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("Add a project name to continue.")
            return
        project = {
            "id": f"p{datetime.now().timestamp()}",
            "name": name.strip(),
            "status": "Draft",
            "description": description.strip() or f"{template} workspace",
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()] or [template],
            "modified": "Just now",
            "files": 0,
            "collaborators": ["A"],
            "cover": "linear-gradient(135deg,#173b63,#793d8c)",
            "icon": icon,
            "starred": False,
            "visibility": visibility,
            "type": "Research",
            "models": 0,
            "pinned": False,
            "accuracy": "—",
            "forecast": ticker,
            "training_enabled": training,
            "auto_sync": auto_sync,
            "invite": invite,
        }
        st.session_state.projects.insert(0, project)
        toast_success("Project created")
        st.rerun()


def _project_card(project: dict[str, object], index: int) -> None:
    """Render a project card and its actions."""
    project_id = str(project["id"])
    tags = " ".join(f'<span class="tag">{tag}</span>' for tag in project["tags"])
    st.markdown(
        f"""
        <div class="project-card">
          <div class="project-cover" style="background:{project["cover"]}"><span class="project-status">{badge(str(project["status"]), str(project["status"]).lower())}</span><span>{project["icon"]}</span><span class="project-star">{"★" if project["starred"] else "☆"} {"•" if project.get("pinned") else ""}</span></div>
          <div class="project-body">
            <div class="project-heading"><div><h3>{project["name"]}</h3><span class="setting-description">{project.get("type", "Analysis")}</span></div></div>
            <p>{project["description"]}</p>
            <div class="tags">{tags}</div>
            <div class="project-meta"><span>{project["modified"]} · {project["files"]} files · {project.get("models", 0)} models</span>{avatar_stack(project["collaborators"])}</div>
            <div class="project-quick-stats"><span>Accuracy <strong>{project.get("accuracy", "—")}</strong></span><span>Forecast <strong>{project.get("forecast", "—")}</strong></span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_cols = st.columns(4)
    actions = [("Open", "open"), ("Share", "share"), ("Duplicate", "duplicate"), ("Delete", "delete")]
    for col, (label, action) in zip(action_cols, actions):
        with col:
            if st.button(label, key=f"project_{action}_{project_id}_{index}", use_container_width=True):
                if action == "duplicate":
                    copy = project.copy()
                    copy["id"] = f"{project_id}-copy"
                    copy["name"] = f'{project["name"]} copy'
                    copy["modified"] = "Just now"
                    st.session_state.projects.insert(0, copy)
                    toast_success("Project duplicated")
                    st.rerun()
                elif action == "delete":
                    st.session_state.confirm_delete = project_id
                    st.rerun()
                else:
                    toast_success(f"{label} action ready")


def _activity_panel() -> None:
    st.markdown('<div class="panel"><div class="panel-title"><h3>Recent activity</h3><span>Team stream</span></div>', unsafe_allow_html=True)
    events = [
        ("Maya updated signal notebook", "12 min ago", "↗"),
        ("LSTM v3 finished training", "41 min ago", "◇"),
        ("You shared Macro Regime Classifier", "2 hr ago", "⇄"),
        ("Ravi added 6 files to Energy RV", "Yesterday", "＋"),
    ]
    for title, time, icon in events:
        st.markdown(f'<div class="timeline-row"><div><span class="event-icon">{icon}</span><span class="ticker">{title}</span></div><small>{time}</small></div>', unsafe_allow_html=True)
    st.markdown("</div><div class=\"section-gap\"></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title"><h3>Your projects</h3><span>Pinned</span></div>', unsafe_allow_html=True)
    for project in [p for p in _projects() if p.get("starred")][:4]:
        st.markdown(f'<div class="quick-project"><span class="project-dot"></span><span>{project["name"]}</span><span class="caption">{project["files"]}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_projects() -> None:
    """Render project search, stats, cards, and activity."""
    projects = _projects()
    top_cols = st.columns([1, 0.22, 0.22, 0.18, 0.18])
    with top_cols[0]:
        st.markdown('<div class="page-kicker">WORKSPACES · 06 TOTAL</div>', unsafe_allow_html=True)
    with top_cols[1]:
        if st.button("＋ New project", type="primary", use_container_width=True):
            _create_project_dialog()
    with top_cols[2]:
        if st.button("Import template", use_container_width=True):
            st.session_state.show_templates = True
    with top_cols[3]:
        if st.button("Import project", use_container_width=True):
            toast_success("Project import flow ready")
    with top_cols[4]:
        if st.button("Sync GitHub", use_container_width=True):
            toast_success("GitHub sync flow ready")
    view_cols = st.columns([1, 0.2])
    with view_cols[1]:
        view = st.selectbox("View", ["Grid", "List", "Timeline", "Kanban"], label_visibility="collapsed")

    search_col, status_col, type_col, owner_col, sort_col = st.columns([1.2, 0.8, 0.8, 0.8, 0.8])
    with search_col:
        query = search_bar("Search projects", "project_query", "Search projects, tags or owners…")
    with status_col:
        status = st.radio("Project filter", ["All", "Active", "Draft", "Archived", "Shared with me"], horizontal=True, label_visibility="collapsed")
    with type_col:
        project_type = st.selectbox("Type", ["All types", "Analysis", "Portfolio", "Model Training", "Research", "Backtest"], label_visibility="collapsed")
    with owner_col:
        owner = st.selectbox("Owner", ["Everyone", "Me", "Team", "Shared"], label_visibility="collapsed")
    with sort_col:
        sort_by = st.selectbox("Sort", ["Recently modified", "Name", "File count", "Star count"], label_visibility="collapsed")

    visible = projects
    if query:
        visible = [project for project in visible if query.lower() in str(project).lower()]
    if status != "All" and status != "Shared with me":
        visible = [project for project in visible if project["status"] == status]
    if project_type != "All types":
        visible = [project for project in visible if project.get("type") == project_type]
    if owner == "Me":
        visible = [project for project in visible if len(project["collaborators"]) == 1]
    if sort_by == "Name":
        visible = sorted(visible, key=lambda project: str(project["name"]))
    elif sort_by == "File count":
        visible = sorted(visible, key=lambda project: int(project["files"]), reverse=True)
    elif sort_by == "Star count":
        visible = sorted(visible, key=lambda project: bool(project["starred"]), reverse=True)

    if st.session_state.get("show_templates"):
        st.markdown('<div class="template-gallery"><div class="panel-title"><h3>Project templates</h3><span>Start with a proven workflow</span></div>', unsafe_allow_html=True)
        template_cols = st.columns(3)
        template_details = [("Stock Analysis", "Equity research with technical indicators.", "12 files · 5 min"), ("Portfolio Tracker", "Holdings, allocation and risk metrics.", "9 files · 4 min"), ("ML Pipeline", "Data, evaluation and deployment flow.", "16 files · 8 min"), ("Research Notebook", "Exploration and report scaffolding.", "7 files · 3 min"), ("Backtest Framework", "Strategy tester and trade log.", "11 files · 6 min")]
        for index, (name, description, setup) in enumerate(template_details):
            with template_cols[index % 3]:
                st.markdown(f'<div class="template-card"><span class="template-icon">{["↗", "▦", "◇", "▤", "ϟ"][index]}</span><strong>{name}</strong><p>{description}</p><span class="caption">{setup}</span></div>', unsafe_allow_html=True)
                if st.button("Use template", key=f"use_template_{name}", use_container_width=True):
                    st.session_state.show_templates = False
                    st.session_state.pending_template = name
                    _create_project_dialog()
        st.markdown("</div><div class=\"section-gap\"></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    stats = st.columns(4)
    with stats[0]:
        stat_card("Total projects", str(len(projects)), "+2 this month", "▦", [2, 2, 3, 3, 5, 6], footer="Across all workspaces")
    with stats[1]:
        stat_card("Active this week", "4", "+1 vs. last week", "↗", [2, 2, 3, 2, 3, 4], footer="Research momentum")
    with stats[2]:
        stat_card("Collaborators", "8", "+3 invited", "◎", [3, 4, 4, 5, 6, 8], footer="Across your projects")
    with stats[3]:
        stat_card("Storage used", "2.4 GB", "24% of 10 GB", "◫", [1, 1.3, 1.7, 1.8, 2.1, 2.4], footer="7.6 GB remaining")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    content_col, activity_col = st.columns([1.8, 0.82], gap="large")
    with content_col:
        st.markdown(f'<div class="panel-title"><h3>{status} projects</h3><span>{len(visible)} results</span></div>', unsafe_allow_html=True)
        if st.session_state.get("confirm_delete"):
            delete_id = st.session_state.confirm_delete
            target = next((project for project in projects if str(project["id"]) == delete_id), None)
            if target:
                st.warning(f'Delete “{target["name"]}”? This removes the project from your workspace.')
                confirm_cols = st.columns([1, 1, 4])
                with confirm_cols[0]:
                    if st.button("Delete permanently", key="confirm_project_delete", type="primary"):
                        st.session_state.projects = [project for project in projects if str(project["id"]) != delete_id]
                        st.session_state.pop("confirm_delete", None)
                        toast_success("Project deleted")
                        st.rerun()
                with confirm_cols[1]:
                    if st.button("Cancel", key="cancel_project_delete"):
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()
        if not visible:
            empty_state("⌕", "No projects found", "Try a different search or create a new analysis workspace.")
        elif view == "Grid":
            card_cols = st.columns(2, gap="medium")
            for index, project in enumerate(visible):
                with card_cols[index % 2]:
                    _project_card(project, index)
                if index % 2 == 1:
                    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        elif view == "List":
            st.dataframe(
                [{"Project": p["name"], "Status": p["status"], "Tags": ", ".join(p["tags"]), "Files": p["files"], "Modified": p["modified"]} for p in visible],
                hide_index=True,
                use_container_width=True,
            )
        elif view == "Timeline":
            for project in visible:
                st.markdown(f'<div class="timeline-project"><span class="timeline-pin"></span><div><strong>{project["name"]}</strong><div class="setting-description">{project["modified"]} · {project["files"]} files · {project["status"]}</div></div></div>', unsafe_allow_html=True)
        else:
            kanban = {"Backlog": [p for p in visible if p["status"] == "Draft"], "In progress": [p for p in visible if p["status"] == "Active"], "Review": [], "Done": [], "Archived": [p for p in visible if p["status"] == "Archived"]}
            board_cols = st.columns(len(kanban))
            for col, (lane, lane_projects) in zip(board_cols, kanban.items()):
                with col:
                    st.markdown(f'<div class="kanban-lane"><div class="kanban-heading"><strong>{lane}</strong><span>{len(lane_projects)}</span></div>', unsafe_allow_html=True)
                    for project in lane_projects:
                        st.markdown(f'<div class="kanban-card"><strong>{project["name"]}</strong><span>{project["files"]} files · {project["type"]}</span></div>', unsafe_allow_html=True)
                    st.markdown('<div class="kanban-add">＋ Add card</div></div>', unsafe_allow_html=True)
    with activity_col:
        _activity_panel()