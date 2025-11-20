import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, brentq, minimize_scalar
import re
import warnings

# --- Configuration & Setup ---
st.set_page_config(
    layout="wide", 
    page_title="Edgeworth Box Simulator",
    page_icon="📊",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings('ignore')

# --- Modern Clean CSS ---
st.markdown("""
<style>
    /* Clean modern color scheme */
    :root {
        --primary: #2E5EAA;
        --secondary: #4A90E2;
        --accent: #F39C12;
        --success: #27AE60;
        --bg-light: #F8F9FA;
        --bg-card: #FFFFFF;
        --text-dark: #2C3E50;
        --text-muted: #7F8C8D;
        --border: #E1E8ED;
    }
    
    /* Main background */
    .stApp {
        background-color: var(--bg-light);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    
    /* Headers */
    h1 {
        color: var(--primary) !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: var(--primary) !important;
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: var(--secondary) !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Cards */
    .info-card {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        background-color: var(--bg-light);
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card);
        color: var(--primary);
    }
    
    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: var(--border);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions (Math) ---
def parse_latex_to_numpy(latex_str):
    if not latex_str: return "0"
    expr = latex_str.lower().replace("^", "**").replace(r"\cdot", "*")
    replacements = {
        r"\\ln": "np.log", r"\\log": "np.log", r"\\exp": "np.exp",
        r"\\sqrt": "np.sqrt", r"\\min": "np.minimum", r"\\max": "np.maximum",
        r"min": "np.minimum", r"max": "np.maximum",
    }
    for tex, py in replacements.items(): expr = re.sub(tex, py, expr)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = re.sub(r'(\d)([xy])', r'\1*\2', expr)
    return expr

def evaluate_custom_utility(x, y, formula):
    try:
        env = {'x': x, 'y': y, 'np': np, 'abs': np.abs, 'log': np.log, 
               'exp': np.exp, 'sqrt': np.sqrt, 'minimum': np.minimum, 'maximum': np.maximum}
        return eval(parse_latex_to_numpy(formula), {"__builtins__": None}, env)
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

def utility_func(x, y, u_type, params):
    x = np.maximum(x, 1e-9)
    y = np.maximum(y, 1e-9)

    if u_type == "Custom (Enter Formula)":
        return evaluate_custom_utility(x, y, params.get('formula', 'x*y'))

    alpha = params.get('alpha', 0.5)
    beta = params.get('beta', 0.5)
    a = params.get('a', 0.0)
    b = params.get('b', 0.0)
    
    if u_type == "Cobb-Douglas":
        return (x ** alpha) * (y ** beta)
    elif u_type == "Perfect Substitutes":
        return alpha * x + beta * y
    elif u_type == "Perfect Complements (Min)": 
        return np.minimum(alpha * x, beta * y)
    elif u_type == "Max Preferences (Convex)": 
        return np.maximum(alpha * x, beta * y)
    elif u_type == "Quasi-Linear (Shifted Product)": 
        return (x + a) * (y + b) 
    elif u_type == "Satiation (Bliss Point)": 
        return -1 * ((x - a)**2 + (y - b)**2)
    elif u_type == "Mixed Cobb-Douglas": 
        return x * (y ** alpha)
    return 0

def calculate_mrs(x, y, u_type, params):
    h = 1e-5
    u0 = utility_func(x, y, u_type, params)
    ux = (utility_func(x + h, y, u_type, params) - u0) / h
    uy = (utility_func(x, y + h, u_type, params) - u0) / h
    
    if abs(uy) < 1e-9:
        if abs(ux) < 1e-9: return 0 
        return np.inf
    return ux / uy

# --- Solver Logic ---
def get_demand(u_type, params, px, py, income, total_x_limit=None, total_y_limit=None):
    """Calculate optimal bundle (x, y) given prices and income."""
    alpha = params.get('alpha', 0.5)
    beta = params.get('beta', 0.5)
    
    if u_type in ["Cobb-Douglas", "Mixed Cobb-Douglas"]:
        if u_type == "Mixed Cobb-Douglas":
            eff_alpha, eff_beta = 1.0, alpha
        else:
            eff_alpha, eff_beta = alpha, beta
            
        x = (eff_alpha / (eff_alpha + eff_beta)) * income / px
        y = (eff_beta / (eff_alpha + eff_beta)) * income / py
        return x, y

    elif u_type == "Perfect Substitutes":
        mrs = alpha / beta
        price_ratio = px / py
        
        if price_ratio < mrs - 1e-6:
            return income / px, 0.0
        elif price_ratio > mrs + 1e-6:
            return 0.0, income / py
        else:
            return income / px, 0.0 

    elif u_type == "Perfect Complements (Min)":
        x = income / (px + py * (alpha / beta))
        y = (alpha / beta) * x
        return x, y

    elif u_type == "Quasi-Linear (Shifted Product)":
        a = params.get('a', 0.0)
        b = params.get('b', 0.0)
        I_eff = income + px*a + py*b
        
        X = I_eff / (2 * px)
        Y = I_eff / (2 * py)
        
        x = max(0, X - a)
        y = (income - px*x) / py
        return x, y

    if u_type == "Max Preferences (Convex)":
        x1, y1 = income / px, 0
        x2, y2 = 0, income / py
        u1 = utility_func(x1, y1, u_type, params)
        u2 = utility_func(x2, y2, u_type, params)
        return (x1, y1) if u1 >= u2 else (x2, y2)

    def obj(v): return -utility_func(v[0], v[1], u_type, params)
    def con_budget(v): return income - (px*v[0] + py*v[1])
    
    b_x = (0, total_x_limit) if total_x_limit else (0, None)
    b_y = (0, total_y_limit) if total_y_limit else (0, None)
    
    x0 = income / (2 * px)
    y0 = income / (2 * py)
    
    res = minimize(obj, [x0, y0], bounds=[b_x, b_y], constraints={'type':'ineq', 'fun':con_budget}, tol=1e-5)
    if res.success:
        return res.x[0], res.x[1]
    
    return x0, y0

def solve_walrasian_equilibrium(total_x, total_y, type_A, params_A, type_B, params_B, endow_A, endow_B):
    py = 1.0
    wAx, wAy = endow_A
    wBx, wBy = endow_B
    
    def excess_demand_x(px):
        if px <= 0: return 1e9 
        IA = px * wAx + py * wAy
        IB = px * wBx + py * wBy
        xA, yA = get_demand(type_A, params_A, px, py, IA, total_x, total_y)
        xB, yB = get_demand(type_B, params_B, px, py, IB, total_x, total_y)
        return (xA + xB) - total_x

    low, high = 0.01, 100.0
    try:
        px_eq = brentq(excess_demand_x, low, high, xtol=1e-4)
    except ValueError:
        res = minimize_scalar(lambda p: abs(excess_demand_x(p)), bounds=(0.01, 100.0), method='bounded')
        px_eq = res.x
    
    IA = px_eq * wAx + py * wAy
    xA, yA = get_demand(type_A, params_A, px_eq, py, IA, total_x, total_y)
    return px_eq, (xA, yA)

def solve_contract_curve(total_x, total_y, type_A, params_A, type_B, params_B, uA_w, uB_w, Z_B_min, Z_B_max):
    pareto_x, pareto_y, core_x, core_y = [], [], [], []
    if Z_B_max <= Z_B_min: return pareto_x, pareto_y, core_x, core_y

    steps = 50
    levels_B = np.linspace(Z_B_min, Z_B_max, steps)
    last_x = [total_x / 2, total_y / 2] 

    for ub_val in levels_B:
        def obj(v): return -utility_func(v[0], v[1], type_A, params_A)
        def con(v): return utility_func(total_x - v[0], total_y - v[1], type_B, params_B) - ub_val
        
        bnds = ((0, total_x), (0, total_y))
        res = minimize(obj, last_x, bounds=bnds, constraints={'type':'ineq', 'fun':con}, tol=1e-5)
        
        best_p = None
        best_u = -np.inf
        
        if res.success:
            best_p = res.x
            last_x = res.x
        else:
            starts = [[0, 0], [total_x, total_y], [0, total_y], [total_x, 0]]
            for s in starts:
                res_retry = minimize(obj, s, bounds=bnds, constraints={'type':'ineq', 'fun':con}, tol=1e-5)
                if res_retry.success:
                    ua = utility_func(res_retry.x[0], res_retry.x[1], type_A, params_A)
                    if ua > best_u:
                        best_u = ua
                        best_p = res_retry.x
                        last_x = res_retry.x

        if best_p is not None:
            ua = utility_func(best_p[0], best_p[1], type_A, params_A)
            ub_real = utility_func(total_x - best_p[0], total_y - best_p[1], type_B, params_B)
            
            if ub_real >= ub_val - 0.1: 
                pareto_x.append(best_p[0])
                pareto_y.append(best_p[1])
                if ua >= uA_w - 1e-3 and ub_val >= uB_w - 1e-3:
                    core_x.append(best_p[0])
                    core_y.append(best_p[1])

    if pareto_x:
        p_points = sorted(zip(pareto_x, pareto_y), key=lambda k: k[0])
        pareto_x, pareto_y = zip(*p_points)
        pareto_x, pareto_y = list(pareto_x), list(pareto_y)

    if core_x:
        c_points = sorted(zip(core_x, core_y), key=lambda k: k[0])
        core_x, core_y = zip(*c_points)
        core_x, core_y = list(core_x), list(core_y)

    return pareto_x, pareto_y, core_x, core_y

# --- Plotting Logic (Plotly) ---
def get_color_scheme():
    """Modern color palette for the visualization"""
    return {
        "A": "#E74C3C",  # Red for Agent A
        "B": "#3498DB",  # Blue for Agent B
        "Pareto": "#2ECC71",  # Green for Pareto set
        "Core": "#F39C12",  # Orange for Core
        "Endowment": "#34495E",  # Dark gray for endowment
        "WE": "#9B59B6",  # Purple for Walrasian Equilibrium
        "Budget": "#7F8C8D",  # Gray for budget line
        "Lens": "rgba(46, 204, 113, 0.15)",  # Transparent green for lens
    }

def plot_edgeworth_box(Z_A, Z_B, x_vec, y_vec, total_x, total_y, 
                       pareto_x, pareto_y, core_x, core_y, 
                       uA_w, uB_w, endow_x, endow_y, 
                       settings, we_data=None):
    
    colors = get_color_scheme()
    
    # Create figure
    fig = go.Figure()
    
    # 1. Exchange Lens (shaded region)
    if settings.get("show_lens", True):
        X, Y = np.meshgrid(x_vec, y_vec)
        lens_mask = np.logical_and(Z_A >= uA_w - 1e-4, Z_B >= uB_w - 1e-4).astype(float)
        lens_mask[lens_mask == 0] = np.nan
        
        fig.add_trace(go.Contour(
            x=x_vec,
            y=y_vec,
            z=lens_mask,
            showscale=False,
            contours=dict(
                start=0.5,
                end=1.5,
                size=1,
                coloring='heatmap'
            ),
            colorscale=[[0, 'rgba(46, 204, 113, 0)'], [1, 'rgba(46, 204, 113, 0.15)']],
            name='Exchange Lens',
            hoverinfo='skip'
        ))
    
    # 2. Indifference Curves for Agent A
    if settings.get("show_curves_A", True):
        n_curves_A = settings.get("n_curves", 20)
        levels_A = np.linspace(np.nanmin(Z_A), np.nanmax(Z_A), n_curves_A)
        
        fig.add_trace(go.Contour(
            x=x_vec,
            y=y_vec,
            z=Z_A,
            contours=dict(
                start=levels_A[0],
                end=levels_A[-1],
                size=(levels_A[-1] - levels_A[0]) / n_curves_A,
            ),
            line=dict(width=1, color=colors["A"]),
            showscale=False,
            name='Agent A Indifference Curves',
            hovertemplate='Agent A<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Utility: %{z:.2f}<extra></extra>'
        ))
    
    # 3. Indifference Curves for Agent B
    if settings.get("show_curves_B", True):
        n_curves_B = settings.get("n_curves", 20)
        levels_B = np.linspace(np.nanmin(Z_B), np.nanmax(Z_B), n_curves_B)
        
        fig.add_trace(go.Contour(
            x=x_vec,
            y=y_vec,
            z=Z_B,
            contours=dict(
                start=levels_B[0],
                end=levels_B[-1],
                size=(levels_B[-1] - levels_B[0]) / n_curves_B,
            ),
            line=dict(width=1, color=colors["B"], dash='dash'),
            showscale=False,
            name='Agent B Indifference Curves',
            hovertemplate='Agent B<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Utility: %{z:.2f}<extra></extra>'
        ))
    
    # 4. Endowment Indifference Curves
    if settings.get("show_endow", True):
        # Agent A endowment curve
        fig.add_trace(go.Contour(
            x=x_vec,
            y=y_vec,
            z=Z_A,
            contours=dict(
                start=uA_w,
                end=uA_w,
                size=1,
            ),
            line=dict(width=3, color=colors["A"]),
            showscale=False,
            name='Initial Utility A',
            hovertemplate='Agent A Initial<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Utility: %{z:.2f}<extra></extra>'
        ))
        
        # Agent B endowment curve
        fig.add_trace(go.Contour(
            x=x_vec,
            y=y_vec,
            z=Z_B,
            contours=dict(
                start=uB_w,
                end=uB_w,
                size=1,
            ),
            line=dict(width=3, color=colors["B"], dash='dash'),
            showscale=False,
            name='Initial Utility B',
            hovertemplate='Agent B Initial<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Utility: %{z:.2f}<extra></extra>'
        ))
    
    # 5. Pareto Set
    if settings.get("show_pareto", True) and pareto_x:
        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode='lines',
            line=dict(color=colors["Pareto"], width=4),
            name='Pareto Set',
            hovertemplate='Pareto Efficient<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    # 6. Core
    if settings.get("show_core", True) and core_x:
        fig.add_trace(go.Scatter(
            x=core_x,
            y=core_y,
            mode='lines',
            line=dict(color=colors["Core"], width=6),
            name='Core',
            hovertemplate='Core<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    # 7. Walrasian Equilibrium
    if settings.get("show_we", False) and we_data:
        px_eq, (xA_eq, yA_eq) = we_data
        
        # Budget Line
        x_range_line = np.array([0, total_x])
        y_line = endow_y + (px_eq / 1.0) * (x_range_line - endow_x)
        
        fig.add_trace(go.Scatter(
            x=x_range_line,
            y=y_line,
            mode='lines',
            line=dict(color=colors["Budget"], width=2, dash='dashdot'),
            name=f'Budget Line (p={px_eq:.2f})',
            hovertemplate='Budget Line<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # WE Point
        fig.add_trace(go.Scatter(
            x=[xA_eq],
            y=[yA_eq],
            mode='markers',
            marker=dict(size=14, color=colors["WE"], symbol='diamond', line=dict(width=2, color='white')),
            name='Walrasian Equilibrium',
            hovertemplate='Walrasian Eq.<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    # 8. Endowment Point
    if settings.get("show_endow", True):
        fig.add_trace(go.Scatter(
            x=[endow_x],
            y=[endow_y],
            mode='markers',
            marker=dict(size=12, color=colors["Endowment"], symbol='circle', line=dict(width=2, color='white')),
            name='Endowment',
            hovertemplate='Endowment<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Edgeworth Box: General Equilibrium Analysis",
            font=dict(size=20, color='#2E5EAA', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Good X (Agent A)',
            range=[-0.5, total_x + 0.5],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            title_font=dict(size=14, color='#2C3E50')
        ),
        yaxis=dict(
            title='Good Y (Agent A)',
            range=[-0.5, total_y + 0.5],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            title_font=dict(size=14, color='#2C3E50'),
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.5)',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#E1E8ED',
            borderwidth=1,
            font=dict(size=11, color='#2C3E50'),
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        width=800,
        height=800,
        margin=dict(l=60, r=200, t=80, b=60)
    )
    
    return fig

# --- Main App (UI) ---

# Initialize Session State for sliders if not exists
def init_state(key, default):
    if key not in st.session_state: st.session_state[key] = default

init_state("dim_x", 10.0)
init_state("dim_y", 10.0)
init_state("endow_x", 7.0)
init_state("endow_y", 6.0)

# App Title
st.title("📊 Edgeworth Box Simulator")
st.markdown("**Interactive general equilibrium analysis with Pareto efficiency and core visualization**")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("🌍 **Environment**", expanded=True):
        st.markdown("*Total endowments in the economy*")
        total_x = st.number_input("Total X", 1.0, 100.0, st.session_state["dim_x"], help="Total quantity of good X")
        total_y = st.number_input("Total Y", 1.0, 100.0, st.session_state["dim_y"], help="Total quantity of good Y")
    
    with st.expander("📦 **Initial Endowment**", expanded=True):
        st.markdown("*Agent A's starting allocation*")
        endow_x = st.slider("Agent A: X", 0.0, total_x, st.session_state["endow_x"])
        endow_y = st.slider("Agent A: Y", 0.0, total_y, st.session_state["endow_y"])
        
        endow_B_x = total_x - endow_x
        endow_B_y = total_y - endow_y
        
        st.caption(f"Agent B gets: X={endow_B_x:.1f}, Y={endow_B_y:.1f}")
    
    with st.expander("🎨 **Display Options**", expanded=False):
        vis_settings = {}
        
        st.markdown("**Equilibrium Concepts**")
        vis_settings["show_pareto"] = st.checkbox("Pareto Set", True)
        vis_settings["show_core"] = st.checkbox("Core", True)
        vis_settings["show_lens"] = st.checkbox("Exchange Lens", True)
        vis_settings["show_we"] = st.checkbox("Walrasian Equilibrium", False)
        
        st.markdown("**Utility Curves**")
        vis_settings["show_endow"] = st.checkbox("Initial Utility Levels", True)
        vis_settings["show_curves_A"] = st.checkbox("Agent A Curves", True)
        vis_settings["show_curves_B"] = st.checkbox("Agent B Curves", True)
        
        vis_settings["n_curves"] = st.slider("Curve Density", 10, 50, 20)

# Main Layout - Preferences and Visualization
st.markdown("### 👥 Agent Preferences")

# Use tabs for cleaner preference input
tab_A, tab_B = st.tabs(["👤 Agent A", "👥 Agent B"])

with tab_A:
    col1, col2 = st.columns([1, 2])
    with col1:
        type_A = st.selectbox(
            "Utility Function Type",
            ["Cobb-Douglas", "Perfect Substitutes", "Perfect Complements (Min)", 
             "Quasi-Linear (Shifted Product)", "Satiation (Bliss Point)"],
            key="type_a"
        )
    with col2:
        params_A = {}
        if type_A == "Cobb-Douglas":
            c1, c2 = st.columns(2)
            with c1:
                params_A["alpha"] = st.slider("α (X weight)", 0.1, 5.0, 1.0, key="alpha_a")
            with c2:
                params_A["beta"] = st.slider("β (Y weight)", 0.1, 5.0, 1.0, key="beta_a")
        elif type_A == "Perfect Substitutes":
            c1, c2 = st.columns(2)
            with c1:
                params_A["alpha"] = st.slider("α (X coefficient)", 0.1, 5.0, 1.0, key="alpha_a_sub")
            with c2:
                params_A["beta"] = st.slider("β (Y coefficient)", 0.1, 5.0, 1.0, key="beta_a_sub")
        elif type_A == "Perfect Complements (Min)":
            c1, c2 = st.columns(2)
            with c1:
                params_A["alpha"] = st.slider("α (X coefficient)", 0.1, 5.0, 1.0, key="alpha_a_comp")
            with c2:
                params_A["beta"] = st.slider("β (Y coefficient)", 0.1, 5.0, 1.0, key="beta_a_comp")
        elif type_A == "Quasi-Linear (Shifted Product)":
            c1, c2 = st.columns(2)
            with c1:
                params_A["a"] = st.slider("X shift (a)", -5.0, 5.0, 0.0, key="a_a")
            with c2:
                params_A["b"] = st.slider("Y shift (b)", -5.0, 5.0, 0.0, key="b_a")
        elif type_A == "Satiation (Bliss Point)":
            c1, c2 = st.columns(2)
            with c1:
                params_A["a"] = st.slider("Bliss X", 0.0, total_x, total_x/2, key="bliss_x_a")
            with c2:
                params_A["b"] = st.slider("Bliss Y", 0.0, total_y, total_y/2, key="bliss_y_a")

with tab_B:
    col1, col2 = st.columns([1, 2])
    with col1:
        type_B = st.selectbox(
            "Utility Function Type",
            ["Cobb-Douglas", "Perfect Substitutes", "Perfect Complements (Min)", 
             "Quasi-Linear (Shifted Product)"],
            key="type_b"
        )
    with col2:
        params_B = {}
        if type_B == "Cobb-Douglas":
            c1, c2 = st.columns(2)
            with c1:
                params_B["alpha"] = st.slider("α (X weight)", 0.1, 5.0, 1.0, key="alpha_b")
            with c2:
                params_B["beta"] = st.slider("β (Y weight)", 0.1, 5.0, 1.0, key="beta_b")
        elif type_B == "Perfect Substitutes":
            c1, c2 = st.columns(2)
            with c1:
                params_B["alpha"] = st.slider("α (X coefficient)", 0.1, 5.0, 1.0, key="alpha_b_sub")
            with c2:
                params_B["beta"] = st.slider("β (Y coefficient)", 0.1, 5.0, 1.0, key="beta_b_sub")
        elif type_B == "Perfect Complements (Min)":
            c1, c2 = st.columns(2)
            with c1:
                params_B["alpha"] = st.slider("α (X coefficient)", 0.1, 5.0, 1.0, key="alpha_b_comp")
            with c2:
                params_B["beta"] = st.slider("β (Y coefficient)", 0.1, 5.0, 1.0, key="beta_b_comp")
        elif type_B == "Quasi-Linear (Shifted Product)":
            c1, c2 = st.columns(2)
            with c1:
                params_B["a"] = st.slider("X shift (a)", -5.0, 5.0, 0.0, key="a_b")
            with c2:
                params_B["b"] = st.slider("Y shift (b)", -5.0, 5.0, 0.0, key="b_b")

st.markdown("---")

# Calculation
N = 100
x_vec = np.linspace(0, total_x, N)
y_vec = np.linspace(0, total_y, N)
X, Y = np.meshgrid(x_vec, y_vec)

Z_A = utility_func(X, Y, type_A, params_A)
if isinstance(Z_A, (float, int)): Z_A = np.full_like(X, Z_A)

Z_B = utility_func(total_x - X, total_y - Y, type_B, params_B)
if isinstance(Z_B, (float, int)): Z_B = np.full_like(X, Z_B)

uA_w = utility_func(endow_x, endow_y, type_A, params_A)
uB_w = utility_func(endow_B_x, endow_B_y, type_B, params_B)

pareto_x, pareto_y, core_x, core_y = solve_contract_curve(
    total_x, total_y, type_A, params_A, type_B, params_B, uA_w, uB_w, np.min(Z_B), np.max(Z_B)
)

we_data = solve_walrasian_equilibrium(
    total_x, total_y, type_A, params_A, type_B, params_B, (endow_x, endow_y), (endow_B_x, endow_B_y)
)

# Visualization
st.markdown("### 📈 Visualization")

fig = plot_edgeworth_box(Z_A, Z_B, x_vec, y_vec, total_x, total_y, 
                         pareto_x, pareto_y, core_x, core_y, 
                         uA_w, uB_w, endow_x, endow_y, 
                         vis_settings, we_data)

st.plotly_chart(fig, use_container_width=True)

# Metrics Section
st.markdown("---")
st.markdown("### 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Agent A Utility", f"{uA_w:.3f}", help="Agent A's utility at endowment")

with col2:
    st.metric("Agent B Utility", f"{uB_w:.3f}", help="Agent B's utility at endowment")

if we_data:
    px, (eq_x, eq_y) = we_data
    with col3:
        st.metric("Equilibrium Price Ratio", f"{px:.3f}", help="Price of X relative to Y")
    with col4:
        uA_eq = utility_func(eq_x, eq_y, type_A, params_A)
        st.metric("Agent A Eq. Utility", f"{uA_eq:.3f}", help="Agent A's utility at equilibrium")
else:
    with col3:
        st.metric("Equilibrium Price Ratio", "—")
    with col4:
        st.metric("Agent A Eq. Utility", "—")

# Information Section
with st.expander("ℹ️ **About This Simulator**"):
    st.markdown("""
    This interactive tool visualizes the Edgeworth Box, a fundamental concept in microeconomic theory 
    for analyzing exchange economies with two agents and two goods.
    
    **Key Concepts:**
    - **Pareto Set (Contract Curve)**: Allocations where no agent can be made better off without making the other worse off
    - **Core**: Allocations that are Pareto efficient and make both agents at least as well off as their initial endowment
    - **Exchange Lens**: Region where both agents can improve upon their initial endowment
    - **Walrasian Equilibrium**: Competitive market equilibrium where supply equals demand at market prices
    
    **How to Use:**
    1. Configure the total endowments in the sidebar
    2. Set the initial endowment for Agent A
    3. Choose utility functions for both agents
    4. Toggle visualization elements to explore different concepts
    """)

