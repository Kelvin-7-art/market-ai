"""Tabbed settings center."""

from __future__ import annotations

import streamlit as st

from market_ai.components.ui.cards import stat_card
from market_ai.components.ui.feedback import toast_success
from market_ai.components.ui.inputs import tab_nav


SETTINGS_TABS = [
    "Profile",
    "Notifications",
    "API Keys",
    "Appearance",
    "Security",
    "Billing",
    "AI & Models",
    "Data Sources",
    "Integrations",
    "Team",
    "Advanced",
    "Help & Support",
]

SETTINGS_ICONS = {
    "Profile": "◉",
    "Notifications": "◌",
    "API Keys": "⌁",
    "Appearance": "◐",
    "Security": "◈",
    "Billing": "◫",
    "AI & Models": "◇",
    "Data Sources": "▦",
    "Integrations": "⌘",
    "Team": "◎",
    "Advanced": "⌬",
    "Help & Support": "?",
}


def _save(label: str = "Save changes") -> None:
    """Render a consistent save action."""
    if st.button(label, type="primary"):
        st.session_state.settings_dirty = False
        toast_success("All changes saved")


@st.dialog("Generate API key")
def _generate_key_dialog() -> None:
    """Generate a clearly fake, frontend-only key preview."""
    name = st.text_input("Key name", placeholder="Research notebook")
    scopes = st.multiselect("Permissions", ["Read", "Write", "Admin"], default=["Read"])
    expiry = st.selectbox("Expires", ["30 days", "90 days", "Never"])
    if st.button("Generate key", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("Add a key name to continue.")
            return
        st.code("mk_demo_••••••••••••••••••••••••", language="text")
        st.caption(f"Demo key · {', '.join(scopes)} · {expiry}. No live credential was created.")
        toast_success("Demo key generated")


def _profile() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">◉</span><div><h3>Profile information</h3><p>How you appear across the research workspace.</p></div></div>', unsafe_allow_html=True)
    avatar_col, form_col = st.columns([0.24, 0.76])
    with avatar_col:
        st.markdown('<div class="profile-avatar">AK<div class="avatar-camera">+</div></div><div class="caption" style="text-align:center;margin-top:.65rem">Aisha Khan</div>', unsafe_allow_html=True)
    with form_col:
        name_col, display_col = st.columns(2)
        name_col.text_input("Full name", "Aisha Khan", key="profile_full_name")
        display_col.text_input("Display name", "Aisha", key="profile_display_name")
        email_col, phone_col = st.columns(2)
        email_col.text_input("Email", "aisha@aiframework.demo", disabled=True, key="profile_email")
        phone_col.text_input("Phone number", "+27 (optional)", key="profile_phone")
        role_col, company_col = st.columns(2)
        role_col.text_input("Job title / role", "Quantitative Researcher", key="profile_role")
        company_col.text_input("Company / organization", "Independent Research", key="profile_company")
        bio = st.text_area("Bio / about", "Quantitative researcher focused on systematic signals and market structure.", height=85, key="profile_bio")
        st.caption(f"{len(bio)}/240 characters")
        tz_col, lang_col, date_col, time_col = st.columns(4)
        tz_col.selectbox("Timezone", ["Africa/Johannesburg", "UTC", "America/New_York"], key="profile_timezone")
        lang_col.selectbox("Language", ["English", "Spanish", "French", "German"], key="profile_language")
        date_col.selectbox("Date format", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"], key="profile_date_format")
        time_col.selectbox("Time format", ["24h", "12h"], key="profile_time_format")
        currency_col, number_col = st.columns(2)
        currency_col.selectbox("Currency", ["USD", "EUR", "GBP", "ZAR"], key="profile_currency")
        number_col.selectbox("Number format", ["1,234.56", "1.234,56"], key="profile_number_format")
        _save()
    st.markdown("</div>", unsafe_allow_html=True)
    actions = st.columns(3)
    for col, label in zip(actions, ["Change email", "Change password", "Download account data"]):
        with col:
            if st.button(label, key=f"profile_action_{label}", use_container_width=True):
                toast_success(f"{label} is ready")
    st.markdown('<div class="danger-zone compact-danger"><strong>Deactivate account</strong><p>Pause access without deleting your research history.</p></div>', unsafe_allow_html=True)


def _notifications() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">◌</span><div><h3>Notification preferences</h3><p>Choose what deserves your attention and where it should arrive.</p></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="subheading">Notification channels</div>', unsafe_allow_html=True)
    channel_cols = st.columns(6)
    for col, channel in zip(channel_cols, ["Email", "In-app", "Push", "Slack", "Discord", "SMS"]):
        with col:
            st.toggle(channel, value=channel in ["Email", "In-app"], key=f"channel_{channel}")
    for category, description in [
        ("Model Events", "Training complete, deploys, failures and accuracy degradation"),
        ("Market Alerts", "Price, volatility, volume and anomaly alerts"),
        ("Forecast Updates", "New forecasts, target hits and confidence changes"),
        ("Sentiment Alerts", "Sentiment shifts, news impact and social spikes"),
        ("System Alerts", "Expiring keys, billing thresholds and source status"),
        ("Team Activity", "New members, shared projects and comments"),
    ]:
        with st.expander(category, expanded=category == "Model Events"):
            st.caption(description)
            cols = st.columns(4)
            for col, channel in zip(cols, ["Email", "In-app", "Push", "Slack"]):
                col.checkbox(channel, value=channel in ["In-app", "Email"], key=f"notif_{category}_{channel}")
            st.multiselect("Events", ["Critical events", "All updates", "Mentions only"], default=["Critical events"], key=f"events_{category}")
            st.selectbox("Frequency", ["Real-time", "Hourly digest", "Daily digest", "Weekly summary"], key=f"notif_freq_{category}")
    delivery_col, quiet_col, priority_col = st.columns(3)
    delivery_col.selectbox("Quiet hours", ["None", "20:00–07:00", "22:00–06:00"], key="quiet_hours")
    quiet_col.toggle("Weekend notifications", value=False, key="weekend_notifications")
    priority_col.selectbox("Priority filter", ["Only critical", "All", "Custom"], key="priority_filter")
    _save()
    st.markdown("</div>", unsafe_allow_html=True)


def _api_keys() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">⌁</span><div><h3>API access</h3><p>Manage programmatic access to your workspace. Live keys are not connected in this demo.</p></div></div>', unsafe_allow_html=True)
    key_action_cols = st.columns([1, 1, 2])
    with key_action_cols[0]:
        if st.button("＋ Generate new key", type="primary", use_container_width=True):
            _generate_key_dialog()
    with key_action_cols[1]:
        if st.button("Add OAuth app", use_container_width=True):
            toast_success("OAuth app flow ready")
    st.dataframe(
        [
            {"Name": "Research notebook", "Prefix": "mk_demo_••••", "Created": "12 Aug 2026", "Last used": "Today", "Permissions": "Read, Write", "Rate limit": "Pro"},
            {"Name": "Dashboard export", "Prefix": "mk_demo_••••", "Created": "28 Jul 2026", "Last used": "3 days ago", "Permissions": "Read", "Rate limit": "Standard"},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("For security, full credentials are only shown once and are never stored in this frontend demo.")
    oauth_col, webhook_col = st.columns(2)
    with oauth_col:
        st.markdown('<div class="subheading">OAuth applications</div>', unsafe_allow_html=True)
        st.markdown('<div class="integration-card"><div class="integration-icon">N</div><div><h4>Notebook sync</h4><p>read:data · write:models</p></div><div class="integration-status active">Authorized</div></div>', unsafe_allow_html=True)
        if st.button("Revoke access", key="revoke_oauth"):
            toast_success("OAuth access revoked")
    with webhook_col:
        st.markdown('<div class="subheading">Webhooks</div>', unsafe_allow_html=True)
        st.markdown('<div class="integration-card"><div class="integration-icon">↗</div><div><h4>Signal delivery</h4><p>3 events · last tested today</p></div><div class="integration-status active">Active</div></div>', unsafe_allow_html=True)
        if st.button("Test webhook", key="test_webhook"):
            toast_success("Webhook test sent")
    st.markdown("</div>", unsafe_allow_html=True)


def _appearance() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">◐</span><div><h3>Appearance</h3><p>Set the density and visual language of your research workspace.</p></div></div>', unsafe_allow_html=True)
    theme = st.radio("Theme", ["Dark", "Light", "System", "Midnight", "Sunset"], horizontal=True, index=0)
    accent = st.select_slider("Accent color", options=["Cyan", "Blue", "Violet", "Emerald", "Amber", "Rose"], value="Cyan")
    custom_color = st.color_picker("Custom accent color", "#31D7EE")
    font_size = st.radio("Font size", ["Small", "Medium", "Large", "Extra large"], horizontal=True, index=1)
    font_family = st.radio("Font family", ["Inter", "System", "Monospace"], horizontal=True, index=0)
    density = st.radio("Density", ["Comfortable", "Compact", "Spacious"], horizontal=True, index=0)
    chart_col, default_col, scheme_col = st.columns(3)
    chart_col.selectbox("Chart style", ["Modern", "Classic", "Minimal"])
    default_col.selectbox("Default chart type", ["Candlestick", "Line", "Area"])
    scheme_col.selectbox("Color scheme", ["Vibrant", "Muted", "Monochrome"])
    st.markdown(f'<div class="appearance-preview"><div class="caption">LIVE PREVIEW · {theme.upper()} · {accent.upper()} · {font_size.upper()} · {density.upper()}</div><div class="preview-line"></div><div class="preview-line short"></div><div class="preview-tile"></div></div>', unsafe_allow_html=True)
    preference_cols = st.columns(4)
    preference_cols.toggle("Animations", value=True)
    preference_cols.toggle("Reduce motion", value=False)
    preference_cols.toggle("High contrast", value=False)
    preference_cols.toggle("Show gridlines", value=True)
    _save()
    st.markdown("</div>", unsafe_allow_html=True)


def _security() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">◈</span><div><h3>Security</h3><p>Protect workspace access and review active sessions.</p></div></div>', unsafe_allow_html=True)
    st.toggle("Two-factor authentication", value=True)
    st.selectbox("Authentication method", ["Authenticator app", "SMS", "Email"])
    st.selectbox("Session timeout", ["After 15 minutes", "After 1 hour", "After 1 day", "Never"])
    security_cols = st.columns(3)
    for col, label in zip(security_cols, ["View backup codes", "Regenerate backup codes", "Sign out all devices"]):
        with col:
            if st.button(label, key=f"security_{label}", use_container_width=True):
                toast_success(f"{label} action ready")
    st.markdown('<div class="subheading">Active sessions</div>', unsafe_allow_html=True)
    st.dataframe([{"Device": "MacBook Pro", "Browser": "Chrome 128", "Location": "Johannesburg, ZA", "IP": "•••.•••.42", "Last active": "Now", "Status": "Current"}, {"Device": "iPhone 15", "Browser": "Safari", "Location": "Johannesburg, ZA", "IP": "•••.•••.91", "Last active": "2 hr ago", "Status": "Active"}], hide_index=True, use_container_width=True)
    with st.expander("Password & privacy"):
        st.text_input("Current password", type="password")
        st.text_input("New password", type="password")
        st.progress(0.72, text="Password strength · Strong")
        st.toggle("Require password change every 90 days", value=False)
        st.toggle("Share anonymous usage analytics", value=True)
        st.toggle("Allow product improvement cookies", value=True)
    st.markdown('<div class="subheading">Login history</div>', unsafe_allow_html=True)
    st.dataframe([{"Event": "Successful login", "Device": "Chrome on macOS", "When": "Today, 09:14"}, {"Event": "Successful login", "Device": "Safari on iPhone", "When": "Yesterday, 18:32"}, {"Event": "Security settings viewed", "Device": "Chrome on macOS", "When": "Yesterday, 16:08"}], hide_index=True, use_container_width=True)
    _save()
    st.markdown("</div>", unsafe_allow_html=True)


def _billing() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">◫</span><div><h3>Plan & usage</h3><p>Track workspace usage and billing details.</p></div></div>', unsafe_allow_html=True)
    stats = st.columns(3)
    with stats[0]:
        stat_card("Current plan", "Research Pro", "Renews 12 Oct", "◇", [1, 1, 1, 1, 1, 1], tone="neutral", footer="$49 / month")
    with stats[1]:
        stat_card("API calls", "68.4k", "68% of monthly limit", "ϟ", [2, 3, 4, 4, 5, 6], footer="31.6k remaining")
    with stats[2]:
        stat_card("Storage", "2.4 GB", "24% of included space", "◫", [1, 1, 2, 2, 2.4, 2.4], footer="10 GB included")
    st.markdown('<div class="usage-grid"><div><div class="usage-label">Training hours <strong>18.4 / 40h</strong></div><div class="progress-track"><div class="progress-fill" style="width:46%"></div></div></div><div><div class="usage-label">Team seats <strong>3 / 5</strong></div><div class="progress-track"><div class="progress-fill" style="width:60%"></div></div></div></div>', unsafe_allow_html=True)
    billing_actions = st.columns(2)
    with billing_actions[0]:
        if st.button("Upgrade plan", type="primary", use_container_width=True):
            toast_success("Plan selection ready")
    with billing_actions[1]:
        if st.button("Manage payment method", use_container_width=True):
            toast_success("Payment method manager ready")
    st.markdown('<div class="payment-method"><div class="integration-icon">▣</div><div><strong>Visa ending in 4242</strong><div class="setting-description">Default · Expires 09/28</div></div><span class="integration-status active">Current</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subheading">Invoice history</div>', unsafe_allow_html=True)
    st.dataframe([{"Invoice": "INV-2026-08", "Date": "12 Aug 2026", "Amount": "$49.00", "Status": "Paid"}, {"Invoice": "INV-2026-07", "Date": "12 Jul 2026", "Amount": "$49.00", "Status": "Paid"}], hide_index=True, use_container_width=True)
    st.markdown('<div class="subheading">Usage analytics</div>', unsafe_allow_html=True)
    st.line_chart({"API calls": [42, 48, 45, 56, 63, 68, 68]}, height=140)
    coupon_col, credits_col = st.columns(2)
    coupon_col.text_input("Promo code", placeholder="Enter code")
    credits_col.number_input("Spending alert threshold ($)", min_value=0, value=75, step=5)
    st.markdown("</div>", unsafe_allow_html=True)


def _integrations() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">⌘</span><div><h3>Integrations</h3><p>Connect the tools and feeds that support your research process.</p></div></div>', unsafe_allow_html=True)
    integrations = [
        ("Interactive Brokers", "Trading platform", "Not connected"),
        ("Alpaca", "Trading platform", "Not connected"),
        ("Binance", "Crypto trading", "Not connected"),
        ("Yahoo Finance", "Market data", "Connected"),
        ("Alpha Vantage", "Market data", "Connected"),
        ("TradingView", "Charting", "Connected"),
        ("Slack", "Notifications", "Not connected"),
        ("Discord", "Notifications", "Not connected"),
        ("Notion", "Productivity", "Not connected"),
        ("GitHub", "Notebook sync", "Not connected"),
        ("Webhooks", "Automation", "Not connected"),
        ("Google Sheets", "Productivity", "Not connected"),
    ]
    cards = st.columns(3)
    for index, (name, description, status) in enumerate(integrations):
        with cards[index % 3]:
            tone = "active" if status == "Connected" else "inactive"
            st.markdown(f'<div class="integration-card"><div class="integration-icon">{name[:1]}</div><div><h4>{name}</h4><p>{description}</p></div><div class="integration-status {tone}">{status}</div></div>', unsafe_allow_html=True)
            if st.button("Manage" if status == "Connected" else "Connect", key=f"integration_{name}", use_container_width=True):
                toast_success(f"{name} action ready")
    st.markdown("</div>", unsafe_allow_html=True)


def _data_sources() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">▦</span><div><h3>Data sources</h3><p>Choose defaults for your market analysis workflows.</p></div></div>', unsafe_allow_html=True)
    source_rows = [
        {"Provider": "Yahoo Finance", "Type": "Market data", "Status": "Connected", "Last sync": "2 min ago"},
        {"Provider": "Alpha Vantage", "Type": "Market data", "Status": "Connected", "Last sync": "5 min ago"},
        {"Provider": "Polygon.io", "Type": "Market data", "Status": "Not connected", "Last sync": "—"},
        {"Provider": "NewsAPI", "Type": "News", "Status": "Not connected", "Last sync": "—"},
    ]
    st.dataframe(source_rows, hide_index=True, use_container_width=True)
    st.multiselect("Default tickers", ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "BTC-USD", "EURUSD"], default=["SPY", "QQQ", "BTC-USD"])
    source_col, interval_col, sentiment_col = st.columns(3)
    source_col.selectbox("Default news source", ["Yahoo News", "NewsAPI", "Reuters", "Bloomberg"])
    interval_col.selectbox("Refresh interval", ["Every 1 minute", "Every 5 minutes", "Every 15 minutes", "Every hour", "Daily"])
    sentiment_col.selectbox("Sentiment model", ["FinBERT", "VADER", "Custom"])
    st.toggle("Enable real-time data", value=False)
    st.toggle("Cache recent market data", value=True)
    st.markdown('<div class="subheading">Cache settings</div>', unsafe_allow_html=True)
    cache_col, clean_col = st.columns(2)
    cache_col.number_input("Cache duration (minutes)", min_value=1, value=15)
    clean_col.toggle("Auto-clean old cache", value=True)
    st.markdown('<div class="subheading">Symbol presets</div>', unsafe_allow_html=True)
    preset_col, preset_action = st.columns([1, 1])
    preset_col.text_input("Symbol preset name", "Core watchlist")
    if preset_action.button("Export watchlist", use_container_width=True):
        toast_success("Watchlist export ready")
    _save()
    st.markdown("</div>", unsafe_allow_html=True)


def _ai_models() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">◇</span><div><h3>AI & model preferences</h3><p>Set defaults for forecasts, feature engineering and the research assistant.</p></div></div>', unsafe_allow_html=True)
    forecast_col, timeframe_col, model_col = st.columns(3)
    forecast_col.selectbox("Default ticker", ["AAPL", "SPY", "QQQ", "BTC-USD", "EURUSD"])
    timeframe_col.selectbox("Default timeframe", ["1D", "1W", "1M", "1Y"])
    model_col.selectbox("Forecasting model", ["Transformer", "LSTM", "XGBoost", "ARIMA"])
    sentiment_col, test_col, cv_col = st.columns(3)
    sentiment_col.selectbox("Sentiment model", ["FinBERT", "VADER", "Custom"])
    test_col.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    cv_col.number_input("CV splits", min_value=2, max_value=10, value=3)
    st.markdown('<div class="subheading">Feature engineering</div>', unsafe_allow_html=True)
    st.multiselect("Technical indicators", ["RSI", "MACD", "Bollinger Bands", "ATR", "EMA", "Volume profile"], default=["RSI", "MACD", "EMA"])
    feature_col, missing_col = st.columns(2)
    feature_col.selectbox("Feature scaling", ["Standard", "MinMax", "None"])
    missing_col.selectbox("Handle missing values", ["Forward fill", "Fill mean", "Fill median", "Drop"])
    st.toggle("Auto-tune hyperparameters", value=True)
    st.number_input("Default forecast days", min_value=1, max_value=365, value=30)
    st.select_slider("Confidence interval", options=["90%", "95%", "99%"], value="95%")
    with st.expander("LLM / chat settings"):
        st.selectbox("Chat model", ["Local demo response", "OpenAI-compatible", "Custom endpoint"])
        st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        st.number_input("Max tokens", min_value=128, max_value=8192, value=2048, step=128)
        st.text_area("System prompt", "You are a careful quantitative research assistant.", height=80)
        st.slider("Memory length", 1, 20, 8)
    _save()
    st.markdown("</div>", unsafe_allow_html=True)


def _advanced() -> None:
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">⌘</span><div><h3>Advanced controls</h3><p>Power-user options for workspace diagnostics.</p></div></div>', unsafe_allow_html=True)
    st.toggle("Developer mode", value=False)
    st.toggle("Show debug information", value=False)
    st.toggle("Show SQL queries", value=False)
    st.toggle("Verbose logging", value=False)
    st.multiselect("Experimental features", ["New LSTM model", "Real-time streaming", "Advanced portfolio optimization", "Multi-asset correlation"], default=["Advanced portfolio optimization"])
    st.markdown('<div class="subheading">Cache management</div>', unsafe_allow_html=True)
    st.markdown('<div class="cache-box"><div><strong>245 MB / 1 GB</strong><div class="setting-description">Last cleared 4 hours ago · application, data and model cache</div></div>', unsafe_allow_html=True)
    cache_cols = st.columns(3)
    for col, label in zip(cache_cols, ["Clear app cache", "Clear data cache", "Clear model cache"]):
        with col:
            if st.button(label, key=f"advanced_{label}", use_container_width=True):
                toast_success(f"{label} complete")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="subheading">Export & import</div>', unsafe_allow_html=True)
    export_cols = st.columns(4)
    for col, label in zip(export_cols, ["Export all data", "Export settings", "Export models", "Export projects"]):
        with col:
            if st.button(label, key=f"export_{label}", use_container_width=True):
                toast_success(f"{label} ready")
    st.markdown('<div class="system-info"><span>TradeGPT v2.0.0</span><span>Build 2026.09.12</span><span>Python 3.12</span><span>Streamlit 1.63</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="danger-zone"><strong>Danger zone</strong><p>Delete account and remove all workspace data.</p>', unsafe_allow_html=True)
    danger_cols = st.columns(3)
    with danger_cols[0]:
        if st.button("Reset all settings", key="reset_settings", type="secondary"):
            st.warning("Reset all settings to defaults?")
    with danger_cols[1]:
        if st.button("Delete all data", key="delete_all_data", type="secondary"):
            st.warning("Delete all data requires confirmation.")
    with danger_cols[2]:
        if st.button("Delete account", key="delete_account", type="secondary"):
            st.warning("Account deletion requires an explicit confirmation in production.")
    st.markdown("</div></div>", unsafe_allow_html=True)


def _team() -> None:
    """Render team membership and collaboration controls."""
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">◎</span><div><h3>Team & collaboration</h3><p>Manage members, roles and shared research resources.</p></div></div>', unsafe_allow_html=True)
    invite_col, role_col = st.columns([1.5, 1])
    with invite_col:
        email = st.text_input("Invite a team member", placeholder="colleague@company.com")
    with role_col:
        role = st.selectbox("Default role", ["Viewer", "Editor", "Admin"])
    if st.button("Send invitation", type="primary"):
        if email.strip():
            toast_success(f"Invitation sent to {email.strip()}")
        else:
            st.warning("Add an email address to continue.")
    st.markdown('<div class="subheading">Team members</div>', unsafe_allow_html=True)
    st.dataframe(
        [
            {"Member": "Aisha Khan", "Email": "aisha@aiframework.demo", "Role": "Owner", "Status": "Active"},
            {"Member": "Maya Patel", "Email": "maya@aiframework.demo", "Role": "Editor", "Status": "Active"},
            {"Member": "Ravi Naidoo", "Email": "ravi@aiframework.demo", "Role": "Viewer", "Status": "Invited"},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown('<div class="subheading">Workspace permissions</div>', unsafe_allow_html=True)
    permission_cols = st.columns(4)
    for col, label in zip(permission_cols, ["Create projects", "Train models", "Deploy models", "Export data"]):
        col.toggle(label, value=label != "Deploy models")
    st.text_input("Organization name", "Independent Research")
    _save()
    st.markdown("</div>", unsafe_allow_html=True)


def _help() -> None:
    """Render help, docs and feedback links."""
    st.markdown('<div class="settings-section"><div class="section-heading"><span class="section-icon">?</span><div><h3>Help & support</h3><p>Find answers, learn the workspace and send feedback.</p></div></div>', unsafe_allow_html=True)
    help_items = [
        ("Quick start guide", "Set up your first project, model and forecast."),
        ("Documentation", "Read the platform reference and API examples."),
        ("Sample projects", "Browse ready-made workspaces for common research flows."),
        ("Video tutorials", "Short walkthroughs for the core research modules."),
        ("Status page", "Check service health and scheduled maintenance."),
        ("Community forum", "Share ideas and learn from other researchers."),
    ]
    cols = st.columns(2)
    for index, (title, description) in enumerate(help_items):
        with cols[index % 2]:
            st.markdown(f'<div class="help-card"><strong>{title}</strong><p>{description}</p></div>', unsafe_allow_html=True)
            if st.button("Open", key=f"help_{title}", use_container_width=True):
                toast_success(f"{title} link ready")
    feedback_col, support_col = st.columns(2)
    with feedback_col:
        if st.button("Report a bug", use_container_width=True):
            toast_success("Bug report form ready")
    with support_col:
        if st.button("Request a feature", use_container_width=True):
            toast_success("Feature request form ready")
    st.markdown('<div class="about-box"><strong>TradeGPT v2.0.0</strong><span>Time Series Intelligence · Built with Streamlit</span><span>Terms · Privacy · Licenses</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_settings() -> None:
    """Render the full settings experience."""
    search_col, reset_col = st.columns([1, 0.18])
    with search_col:
        settings_query = st.text_input("Search settings", placeholder="Search settings…", label_visibility="collapsed", key="settings_query")
    with reset_col:
        if st.button("Reset", use_container_width=True):
            st.session_state.settings_dirty = False
            toast_success("Settings reset to defaults")
    st.markdown('<div class="settings-status"><span class="status-dot"></span> All changes saved <span>· Autosave is on</span></div>', unsafe_allow_html=True)
    nav_col, content_col = st.columns([0.28, 0.72], gap="large")
    with nav_col:
        st.markdown('<div class="panel settings-nav"><div class="caption">WORKSPACE SETTINGS</div>', unsafe_allow_html=True)
        matching_tabs = [tab for tab in SETTINGS_TABS if not settings_query or settings_query.lower() in tab.lower()]
        if not matching_tabs:
            st.caption("No matching sections")
            matching_tabs = SETTINGS_TABS
        tab = tab_nav("Settings navigation", matching_tabs, "settings_tab")
        st.markdown("</div>", unsafe_allow_html=True)
    with content_col:
        if tab == "Profile":
            _profile()
        elif tab == "Notifications":
            _notifications()
        elif tab == "API Keys":
            _api_keys()
        elif tab == "Appearance":
            _appearance()
        elif tab == "Security":
            _security()
        elif tab == "Billing":
            _billing()
        elif tab == "AI & Models":
            _ai_models()
        elif tab == "Team":
            _team()
        elif tab == "Integrations":
            _integrations()
        elif tab == "Data Sources":
            _data_sources()
        elif tab == "Help & Support":
            _help()
        else:
            _advanced()