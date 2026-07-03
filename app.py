from io import StringIO
from pathlib import Path
from copy import deepcopy
from urllib.request import urlopen

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from branca.colormap import linear
from streamlit_folium import st_folium


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://github.com/luizgustavogr/dashboard-datatran/releases/download/v1.0/datatran_unificado.csv"
LOCAL_DATA_FILE = BASE_DIR / "datatran_unificado.csv"
PLACAS_DATA_FILE = BASE_DIR / "placas_sinalizacao_processado.csv"
RADARES_DATA_FILE = BASE_DIR / "radares_velocidade_processado.csv"
MOCK_END_YEAR = 2023
BRAZIL_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
USECOLS = [
    "id",
    "data_inversa",
    "dia_semana",
    "br",
    "km",
    "uf",
    "municipio",
    "causa_acidente",
    "classificacao_acidente",
    "condicao_metereologica",
    "fase_dia",
    "tipo_pista",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "feridos",
    "veiculos",
]

# Cores padrão para o contexto de acidentes (Cores quentes para casos mais graves)
COLOR_ACIDENTES = "#1D4ED8"     # Azul escuro/royal para acidentes (neutro)
COLOR_MORTOS = "#991B1B"        # Vermelho escuro/carmesim para casos fatais (grave)
COLOR_FERIDOS_LEVES = "#FBBF24" # Amarelo/âmbar para ferimentos leves
COLOR_FERIDOS = "#F97316"       # Laranja para ferimentos gerais
COLOR_FERIDOS_GRAVES = "#EF4444" # Vermelho claro/laranja-avermelhado para ferimentos graves

COLOR_MAP_SEVERIDADE = {
    "Feridos leves": COLOR_FERIDOS_LEVES,
    "Feridos": COLOR_FERIDOS,
    "Feridos graves": COLOR_FERIDOS_GRAVES,
    "Mortos": COLOR_MORTOS
}

#teste
st.set_page_config(
    page_title="Dashboard DATATRAN",
    page_icon="📊",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def load_data(source: str | Path) -> pd.DataFrame:
    read_kwargs = {
        "sep": ";",
        "low_memory": False,
        "usecols": lambda column: column in USECOLS,
    }

    if isinstance(source, Path):
        df = pd.read_csv(source, encoding="utf-8", **read_kwargs)
    else:
        with urlopen(source) as response:
            csv_text = response.read().decode("utf-8-sig")
        df = pd.read_csv(StringIO(csv_text), **read_kwargs)

    df.columns = [column.strip().lower() for column in df.columns]

    if "data_inversa" not in df.columns:
        raise ValueError("A coluna data_inversa nao foi encontrada no CSV unificado.")

    df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    df["ano"] = df["data_inversa"].dt.year

    text_columns = ["uf", "municipio", "causa_acidente", "tipo_acidente", "classificacao_acidente", "condicao_metereologica", "fase_dia", "tipo_pista", "dia_semana"]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    if "dia_semana" not in df.columns:
        dias_semana = {
            0: "segunda-feira",
            1: "terça-feira",
            2: "quarta-feira",
            3: "quinta-feira",
            4: "sexta-feira",
            5: "sábado",
            6: "domingo",
        }
        df["dia_semana"] = df["data_inversa"].dt.dayofweek.map(dias_semana)

    numeric_columns = [
        "pessoas",
        "mortos",
        "feridos_leves",
        "feridos_graves",
        "feridos",
        "ilesos",
        "ignorados",
        "veiculos",
        "km",
        "latitude",
        "longitude",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def format_number(value: float) -> str:
    if pd.isna(value):
        return "0"
    return f"{int(value):,}".replace(",", ".")


@st.cache_data(show_spinner=False)
def load_brazil_states_geojson() -> dict:
    response = requests.get(BRAZIL_STATES_GEOJSON_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def _point_in_ring(lng: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False

    previous_x, previous_y = ring[-1]
    for current_x, current_y in ring:
        intersects = ((current_y > lat) != (previous_y > lat)) and (
            lng < (previous_x - current_x) * (lat - current_y) / ((previous_y - current_y) or 1e-12) + current_x
        )
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_in_geometry(lng: float, lat: float, geometry: dict) -> bool:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    def polygon_contains(polygon_coordinates: list[list[list[float]]]) -> bool:
        if not polygon_coordinates:
            return False
        if not _point_in_ring(lng, lat, polygon_coordinates[0]):
            return False
        return not any(_point_in_ring(lng, lat, hole) for hole in polygon_coordinates[1:])

    if geometry_type == "Polygon":
        return polygon_contains(coordinates)

    if geometry_type == "MultiPolygon":
        return any(polygon_contains(polygon) for polygon in coordinates)

    return False


def _geometry_centroid(geometry: dict) -> tuple[float, float] | None:
    points: list[tuple[float, float]] = []

    def collect(coords: list) -> None:
        for item in coords:
            if isinstance(item[0], (int, float)):
                points.append((float(item[0]), float(item[1])))
            else:
                collect(item)

    collect(geometry.get("coordinates", []))
    if not points:
        return None

    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return (sum(longitudes) / len(longitudes), sum(latitudes) / len(latitudes))


def build_brazil_map(uf_summary: pd.DataFrame, selected_ufs: list[str]) -> folium.Map:
    geojson_data = load_brazil_states_geojson()
    enriched_geojson = deepcopy(geojson_data)
    summary_by_uf = {
        str(row["uf"]): row for _, row in uf_summary.iterrows()
    }

    if uf_summary.empty:
        min_accidents = 0
        max_accidents = 1
    else:
        min_accidents = int(uf_summary["acidentes"].min())
        max_accidents = int(uf_summary["acidentes"].max())
        if min_accidents == max_accidents:
            max_accidents = min_accidents + 1

    colormap = linear.YlOrRd_09.scale(min_accidents, max_accidents)
    # colormap.caption = "Número de acidentees"

    for feature in enriched_geojson.get("features", []):
        properties = feature.setdefault("properties", {})
        sigla = str(properties.get("sigla", "")).strip()
        summary_row = summary_by_uf.get(sigla)
        acidentes = int(summary_row["acidentes"]) if summary_row is not None else 0
        mortos = float(summary_row["mortos"]) if summary_row is not None else 0.0
        letalidade = (mortos / acidentes * 100) if acidentes else 0.0

        properties["uf"] = sigla
        properties["nome_estado"] = properties.get("name", sigla)
        properties["acidentes"] = acidentes
        properties["letalidade"] = f"{letalidade:.2f}%"
        properties["selecionado"] = sigla in selected_ufs

    brazil_map = folium.Map(location=[-14.235, -51.9253], zoom_start=4, tiles="cartodbpositron", control_scale=True)

    def style_function(feature: dict) -> dict:
        properties = feature.get("properties", {})
        acidentes = int(properties.get("acidentes", 0))
        selecionado = bool(properties.get("selecionado", False))
        base_fill = "#FFFDF5" if acidentes == 0 else colormap(acidentes)

        if selecionado:
            return {
                "fillColor": "#7F1D1D",
                "color": "#000000",
                "weight": 3.5,
                "fillOpacity": 0.85,
            }

        return {
            "fillColor": base_fill,
            "color": "#D97706",
            "weight": 1.3,
            "fillOpacity": 0.72,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["nome_estado", "uf", "acidentes", "letalidade"],
        aliases=["Nome do estado:", "Sigla do estado:", "Número de acidentes:", "Taxa de letalidade:"],
        localize=True,
        sticky=False,
        labels=True,
        style=(
            "background-color: white; color: #111827; font-family: Arial; font-size: 13px; padding: 8px;"
        ),
        max_width=260,
    )

    geojson_layer = folium.GeoJson(
        enriched_geojson,
        name="Estados",
        style_function=style_function,
        highlight_function=lambda feature: {
            "weight": 3.5,
            "color": "#B91C1C",
            "fillOpacity": 0.9,
        },
        tooltip=tooltip,
    )
    geojson_layer.add_to(brazil_map)

    try:
        brazil_map.fit_bounds(geojson_layer.get_bounds())
    except Exception:
        pass

    return brazil_map


def resolve_clicked_uf(click_data: dict, geojson_data: dict) -> str | None:
    lat = click_data.get("lat")
    lng = click_data.get("lng")
    if lat is None or lng is None:
        return None

    for feature in geojson_data.get("features", []):
        properties = feature.get("properties", {})
        sigla = str(properties.get("sigla", "")).strip()
        geometry = feature.get("geometry", {})
        if sigla and _point_in_geometry(float(lng), float(lat), geometry):
            return sigla

    return None


def build_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("ano", dropna=True)
        .agg(
            acidentes=("id", "count"),
            pessoas=("pessoas", "sum"),
            mortos=("mortos", "sum"),
            feridos_leves=("feridos_leves", "sum"),
            feridos_graves=("feridos_graves", "sum"),
            feridos=("feridos", "sum"),
            veiculos=("veiculos", "sum"),
        )
        .reset_index()
        .sort_values("ano")
    )

    summary["feridos_totais"] = summary["feridos"].fillna(0)
    summary["mortes_por_1000_acidentes"] = (
        summary["mortos"].fillna(0) / summary["acidentes"].replace(0, pd.NA) * 1000
    ).fillna(0)
    summary["variacao_acidentes_pct"] = summary["acidentes"].pct_change().mul(100).fillna(0)
    summary["variacao_mortos_pct"] = summary["mortos"].pct_change().mul(100).fillna(0)

    return summary


def normalize_br_series(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def merge_intervals(intervals: pd.DataFrame) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in intervals.sort_values("km_m_inicio")[["km_m_inicio", "km_m_final"]].to_numpy(dtype=float):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


@st.cache_data(show_spinner=False)
def build_plate_match_summary(datatran_source: str | Path, placas_source: str | Path) -> pd.DataFrame:
    accident_df = pd.read_csv(
        datatran_source,
        sep=";",
        low_memory=False,
        encoding="utf-8-sig",
        usecols=lambda column: column in {"id", "data_inversa", "br", "km"},
    )
    accident_df.columns = [column.strip().lower() for column in accident_df.columns]
    accident_df["data_inversa"] = pd.to_datetime(accident_df["data_inversa"], errors="coerce")
    accident_df["ano"] = accident_df["data_inversa"].dt.year
    accident_df["br"] = normalize_br_series(accident_df.get("br", pd.Series(dtype="string")))
    accident_df["km"] = pd.to_numeric(accident_df.get("km", pd.Series(dtype="float")), errors="coerce")
    accident_df = accident_df.dropna(subset=["ano", "br", "km"])
    accident_df["ano"] = accident_df["ano"].astype(int)

    placas_df = pd.read_csv(placas_source, sep=";", low_memory=False, encoding="utf-8-sig")
    placas_df.columns = [column.strip().lower() for column in placas_df.columns]
    placas_df["br"] = pd.to_numeric(placas_df.get("br"), errors="coerce").astype("Int64")
    placas_df["km_m_inicio"] = pd.to_numeric(placas_df.get("km_m_inicio"), errors="coerce")
    placas_df["km_m_final"] = pd.to_numeric(placas_df.get("km_m_final"), errors="coerce")
    placas_df = placas_df.dropna(subset=["br", "km_m_inicio", "km_m_final"])

    plates_by_br = {
        int(br): merge_intervals(group)
        for br, group in placas_df.groupby("br", dropna=True)
    }

    accident_df["placa_encontrada"] = False
    for br_value, accident_group in accident_df.groupby("br", sort=False):
        intervals = plates_by_br.get(int(br_value))
        if not intervals:
            continue

        starts = pd.Series([start for start, _ in intervals], dtype="float64").to_numpy()
        ends = pd.Series([end for _, end in intervals], dtype="float64").to_numpy()
        km_values = accident_group["km"].to_numpy(dtype=float)

        interval_index = starts.searchsorted(km_values, side="right") - 1
        valid = interval_index >= 0
        valid_km = km_values[valid]
        valid_index = interval_index[valid]
        matched = valid_km <= ends[valid_index]

        accident_df.loc[accident_group.index, "placa_encontrada"] = False
        accident_df.loc[accident_group.index[valid], "placa_encontrada"] = matched

    summary = (
        accident_df[accident_df["ano"] <= MOCK_END_YEAR]
        .groupby(["ano", "placa_encontrada"], dropna=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    summary = summary.rename(columns={True: "Com placa", False: "Sem placa"})
    if "Com placa" not in summary.columns:
        summary["Com placa"] = 0
    if "Sem placa" not in summary.columns:
        summary["Sem placa"] = 0

    summary = summary.rename(columns={"ano": "Ano"}).sort_values("Ano")
    return summary


@st.cache_data(show_spinner=False, ttl=3600)
def load_radar_data() -> pd.DataFrame:
    try:
        url = "https://servicos.dnit.gov.br/services-sior/portal-multas/pncv/equipamentos"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        geojson_data = response.json()
        features = geojson_data.get("features", [])
        records = []
        for feature in features:
            prop = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            records.append({
                "uf": str(prop.get("uf", "")).strip().upper(),
                "municipio": str(prop.get("municipio", "")).strip().upper(),
                "br": str(prop.get("rodovia", "")).strip(),
                "km": str(prop.get("km", "")).strip(),
                "latitude": coords[1],
                "longitude": coords[0],
                "tipo": str(prop.get("tipoNome", "")).strip(),
            })
        return pd.DataFrame(records)
    except Exception:
        # Fallback to local file if available
        if Path(RADARES_DATA_FILE).exists():
            try:
                df = pd.read_csv(RADARES_DATA_FILE, sep=";")
                df.columns = [col.strip().lower() for col in df.columns]
                if "uf" in df.columns:
                    df["uf"] = df["uf"].astype(str).str.strip().str.upper()
                return df
            except Exception:
                pass
        return pd.DataFrame()


def apply_sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    st.sidebar.header("Filtros")

    years = sorted([int(year) for year in df["ano"].dropna().unique()])
    selected_years = st.sidebar.multiselect("Faixa de anos", years, default=years)

    # Filtra por classificação primeiro nos dados originais para termos o dataframe base sem filtro de anos
    classification_filtered = df
    if "classificacao_acidente" in df.columns:
        classificacoes = sorted(df["classificacao_acidente"].dropna().astype(str).unique().tolist())
        selected_classificacoes = st.sidebar.multiselect("Classificação", classificacoes, default=classificacoes)
        if selected_classificacoes:
            classification_filtered = df[df["classificacao_acidente"].isin(selected_classificacoes)]

    # Aplica filtro de anos para o dataframe principal
    if selected_years:
        filtered = classification_filtered[classification_filtered["ano"].isin(selected_years)]
    else:
        filtered = classification_filtered.iloc[0:0]

    return filtered, classification_filtered, selected_years


def add_metric_card(label: str, value: float | str, delta: str | None = None, delta_color: str = "normal") -> None:
    formatted_value = value if isinstance(value, str) else format_number(value)
    st.metric(label, formatted_value, delta, delta_color=delta_color, border=True)


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "taxa_gravidade": 0.0,
            "taxa_chuva": 0.0,
            "taxa_mortalidade": 0.0
        }
    
    # 1. Taxa de gravidade em pistas simples (acidentes graves em pistas simples / total de acidentes em pistas simples)
    pistas_simples = df[df["tipo_pista"].fillna("").astype(str).str.strip().str.capitalize() == "Simples"]
    total_simples = pistas_simples.shape[0]
    graves_simples = pistas_simples[
        (pistas_simples["feridos_graves"].fillna(0) > 0) | (pistas_simples["mortos"].fillna(0) > 0)
    ].shape[0]
    taxa_gravidade = (graves_simples / total_simples * 100) if total_simples else 0.0

    # 2. Proporção de acidentes em condições de chuva (acidentes sob chuva / total de acidentes)
    total_acidentes = df.shape[0]
    acidentes_chuva = df[df["condicao_metereologica"].fillna("").astype(str).str.strip().str.lower().str.contains("chuva", na=False)].shape[0]
    taxa_chuva = (acidentes_chuva / total_acidentes * 100) if total_acidentes else 0.0

    # 3. Taxa de mortalidade (óbitos / total de acidentes)
    mortos_totais = df["mortos"].fillna(0).sum()
    taxa_mortalidade = (mortos_totais / total_acidentes * 100) if total_acidentes else 0.0

    return {
        "taxa_gravidade": taxa_gravidade,
        "taxa_chuva": taxa_chuva,
        "taxa_mortalidade": taxa_mortalidade
    }


def main() -> None:
    # Custom CSS for elevated (popped-up) metric cards
    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
            background-color: #f1f5f9;
            border: 2px solid #cbd5e1;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.06);
            border-color: #94a3b8;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Dashboard DATATRAN")
    st.caption("Análise de dados sobre acidentes em rodovias federais brasileiras.")

    try:
        df = load_data(DATA_URL)
    except Exception:
        st.error("Não foi possível carregar os dados.")
        st.stop()

    df = df.dropna(subset=["ano"])

    # 1. Apply Sidebar Filters
    sidebar_filtered, classification_filtered, selected_years = apply_sidebar_filters(df)

    # 2. Regional Map filter state (single selection)
    selected_ufs_key = "selected_ufs_map"
    if selected_ufs_key not in st.session_state:
        st.session_state[selected_ufs_key] = []

    # Clean up selection if not in available UFs of the current sidebar selection
    available_ufs = sidebar_filtered["uf"].dropna().unique().tolist()
    st.session_state[selected_ufs_key] = [uf for uf in st.session_state[selected_ufs_key] if uf in available_ufs]

    # Calculate final filtered DataFrame (including regional state selection)
    selected_ufs = st.session_state[selected_ufs_key]
    if selected_ufs:
        filtered = sidebar_filtered[sidebar_filtered["uf"].isin(selected_ufs)]
    else:
        filtered = sidebar_filtered

    if filtered.empty:
        st.warning("Nenhum registro corresponde aos filtros selecionados.")
        st.stop()

    # Load Speed Radar Data from DNIT API
    radar_df = load_radar_data()
    if not radar_df.empty:
        if selected_ufs:
            selected_uf = selected_ufs[0]
            radar_count = radar_df[radar_df["uf"] == selected_uf.upper()].shape[0]
        else:
            radar_count = radar_df.shape[0]
    else:
        radar_count = 0

    # 3. KPIs de Segurança Viária com deltas baseados nas regras do usuário
    if selected_years:
        sorted_years = sorted([int(y) for y in selected_years])
        year_curr = sorted_years[-1]
        
        # Regras de comparação para o delta:
        if len(sorted_years) >= 2:
            # Se filtrar por dois ou mais anos (ou todos os anos), compara o mais recente com o mais próximo
            year_prev = sorted_years[-2]
        else:
            # Se tiver somente um, compara com o anterior (calendário)
            year_prev = year_curr - 1
            # Se for o primeiro ano do dataset geral, não há o que mostrar
            min_possible_year = int(df["ano"].dropna().min())
            if year_prev < min_possible_year:
                year_prev = None
    else:
        year_curr = None
        year_prev = None

    # Slices de dados para o cálculo (aplicando o mesmo filtro de UF selecionada se houver)
    if selected_ufs:
        df_curr_base = classification_filtered[classification_filtered["uf"].isin(selected_ufs)]
    else:
        df_curr_base = classification_filtered

    # Os KPIs mostram o valor do ano mais recente (year_curr) e o delta compara com o anterior (year_prev)
    if year_curr is not None:
        df_curr = df_curr_base[df_curr_base["ano"] == year_curr]
    else:
        df_curr = df_curr_base.iloc[0:0]

    if year_prev is not None:
        df_prev = df_curr_base[df_curr_base["ano"] == year_prev]
    else:
        df_prev = df_curr_base.iloc[0:0]

    kpis_curr = calculate_kpis(df_curr)
    kpis_prev = calculate_kpis(df_prev)

    # Formatação de delta de forma relativa
    def format_kpi_delta(val_curr: float, val_prev: float) -> str | None:
        if year_prev is None or not val_prev:
            return None
        pct_change = ((val_curr - val_prev) / val_prev * 100)
        return f"{pct_change:+.2f}%"

    delta_gravidade = format_kpi_delta(kpis_curr["taxa_gravidade"], kpis_prev["taxa_gravidade"])
    delta_chuva = format_kpi_delta(kpis_curr["taxa_chuva"], kpis_prev["taxa_chuva"])
    delta_mortalidade = format_kpi_delta(kpis_curr["taxa_mortalidade"], kpis_prev["taxa_mortalidade"])

    st.subheader("Principais Métricas")
    if year_curr is not None:
        if year_prev is not None:
            caption_text = f"Métricas calculadas para o ano de {year_curr}. Os deltas comparam a variação percentual relativa contra o ano de {year_prev} (redução = verde)."
        else:
            caption_text = f"Métricas calculadas para o ano inicial de {year_curr} (sem comparação anterior)."
    else:
        caption_text = "Selecione pelo menos um ano para calcular as métricas."
    st.caption(caption_text)

    kpi_cols = st.columns(3)

    with kpi_cols[0]:
        add_metric_card(
            label="Taxa de Gravidade (Pista Simples)",
            value=f"{kpis_curr['taxa_gravidade']:.2f}%",
            delta=delta_gravidade,
            delta_color="inverse"
        )
    with kpi_cols[1]:
        add_metric_card(
            label="Taxa de Acidentes sob Chuva",
            value=f"{kpis_curr['taxa_chuva']:.2f}%",
            delta=delta_chuva,
            delta_color="inverse"
        )
    with kpi_cols[2]:
        add_metric_card(
            label="Taxa de Mortalidade",
            value=f"{kpis_curr['taxa_mortalidade']:.2f}%",
            delta=delta_mortalidade,
            delta_color="inverse"
        )

    # 3.1 Metric cards
    st.subheader("Panorama Geral")
    metric_cols = st.columns(5)

    with metric_cols[0]:
        add_metric_card("Acidentes", filtered.shape[0])
    with metric_cols[1]:
        add_metric_card("Mortos", filtered["mortos"].fillna(0).sum())
    with metric_cols[2]:
        add_metric_card("Feridos totais", filtered["feridos"].fillna(0).sum())
    with metric_cols[3]:
        add_metric_card("Veículos", filtered["veiculos"].fillna(0).sum())
    with metric_cols[4]:
        add_metric_card("Radares de Velocidade (DNIT)", radar_count)

    st.divider()

    # 4. Regional Panorama (3-column layout)
    st.subheader("Panorama Regional")
    
    col_left, col_mid, col_right = st.columns((1.1, 1.8, 1.1))
    
    # Left Column: Top 5 States
    with col_left:
        top_states = (
            sidebar_filtered.groupby("uf", dropna=True)
            .agg(acidentes=("id", "count"))
            .reset_index()
            .sort_values("acidentes", ascending=False)
            .head(5)
        )
        fig_top_states = px.bar(
            top_states,
            x="uf",
            y="acidentes",
            color="acidentes",
            color_continuous_scale=["#FDBA74", "#EF4444", "#991B1B"], # Cores quentes baseadas em intensidade
        )
        fig_top_states.update_layout(
            title="Top 5 Estados",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
            xaxis_title="UF",
            yaxis_title="Acidentes",
        )
        st.plotly_chart(fig_top_states, use_container_width=True)

    # Middle Column: Map
    with col_mid:
        st.markdown("<p style='text-align: center; font-size: 0.85rem; color: gray; margin-bottom: 5px;'>Clique em um estado para filtrar. Clique novamente para limpar.</p>", unsafe_allow_html=True)
        
        uf_summary_map = (
            sidebar_filtered.dropna(subset=["uf"])
            .assign(uf=lambda frame: frame["uf"].astype("string").str.strip())
            .groupby("uf", dropna=True)
            .agg(acidentes=("id", "count"), mortos=("mortos", "sum"))
            .reset_index()
            .sort_values("acidentes", ascending=False)
        )
        
        brazil_map = build_brazil_map(uf_summary_map, st.session_state[selected_ufs_key])
        map_state = st_folium(
            brazil_map,
            key="brazil_map",
            height=350,
            use_container_width=True,
            returned_objects=["last_object_clicked"],
        )

    # Right Column: Top 5 Municipalities
    with col_right:
        selected_uf = st.session_state[selected_ufs_key][0] if st.session_state[selected_ufs_key] else None
        
        if selected_uf:
            muni_data = sidebar_filtered[sidebar_filtered["uf"] == selected_uf]
            title_muni = f"Top 5 Municípios ({selected_uf})"
        else:
            muni_data = sidebar_filtered
            title_muni = "Top 5 Municípios (Brasil)"
            
        top_munis = (
            muni_data.dropna(subset=["municipio"])
            .assign(municipio=lambda frame: frame["municipio"].astype("string").str.strip())
            .groupby("municipio", dropna=True)
            .agg(acidentes=("id", "count"))
            .reset_index()
            .sort_values("acidentes", ascending=False)
            .head(5)
        )
        
        fig_top_munis = px.bar(
            top_munis,
            x="municipio",
            y="acidentes",
            color="acidentes",
            color_continuous_scale=["#FDBA74", "#EF4444", "#991B1B"], # Cores quentes baseadas em intensidade
        )
        fig_top_munis.update_layout(
            title=title_muni,
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
            xaxis_title="Município",
            yaxis_title="Acidentes",
        )
        fig_top_munis.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_top_munis, use_container_width=True)

    # Handle map clicks (single selection)
    geojson_data = load_brazil_states_geojson()
    clicked = map_state.get("last_object_clicked") if isinstance(map_state, dict) else None
    if clicked:
        clicked_uf = resolve_clicked_uf(clicked, geojson_data)
        if clicked_uf:
            if clicked_uf in st.session_state[selected_ufs_key]:
                st.session_state[selected_ufs_key] = []
            else:
                st.session_state[selected_ufs_key] = [clicked_uf]
            st.rerun()

    st.divider()

    # 5. Comparativo entre Anos (Não é afetado pelo filtro de faixa de anos da sidebar)
    if selected_ufs:
        year_filtered_df = classification_filtered[classification_filtered["uf"].isin(selected_ufs)]
    else:
        year_filtered_df = classification_filtered

    year_summary = build_year_summary(year_filtered_df)
    
    # Antiga implementação de gráfico único comentado
    # st.subheader("Comparativo entre anos")
    # fig_years = go.Figure()
    # fig_years.add_trace(
    #     go.Scatter(
    #         x=year_summary["ano"],
    #         y=year_summary["acidentes"],
    #         name="Acidentes",
    #         mode="lines+markers",
    #         line=dict(color="#2E86DE", width=3),
    #     )
    # )
    # fig_years.add_trace(
    #     go.Scatter(
    #         x=year_summary["ano"],
    #         y=year_summary["mortos"],
    #         name="Mortos",
    #         mode="lines+markers",
    #         line=dict(color="#C0392B", width=3),
    #     )
    # )
    # fig_years.update_layout(
    #     height=450,
    #     xaxis_title="Ano",
    #     yaxis_title="Quantidade",
    #     legend_title_text="Série",
    #     margin=dict(l=10, r=10, t=30, b=10),
    # )
    # st.plotly_chart(fig_years, use_container_width=True)

    # Nova implementação com dois gráficos separados lado a lado
    st.subheader("Comparativo entre anos")
    col_years_1, col_years_2 = st.columns(2)
    
    with col_years_1:
        st.markdown("<h5 style='text-align: center;'>Total de Acidentes</h5>", unsafe_allow_html=True)
        fig_accidents = go.Figure()
        fig_accidents.add_trace(
            go.Scatter(
                x=year_summary["ano"],
                y=year_summary["acidentes"],
                name="Acidentes",
                mode="lines+markers",
                line=dict(color=COLOR_ACIDENTES, width=3),
            )
        )
        fig_accidents.update_layout(
            height=380,
            xaxis_title="Ano",
            yaxis_title="Quantidade",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_accidents, use_container_width=True)
        
    with col_years_2:
        st.markdown("<h5 style='text-align: center;'>Total de Mortos</h5>", unsafe_allow_html=True)
        fig_deaths = go.Figure()
        fig_deaths.add_trace(
            go.Scatter(
                x=year_summary["ano"],
                y=year_summary["mortos"],
                name="Mortos",
                mode="lines+markers",
                line=dict(color=COLOR_MORTOS, width=3),
            )
        )
        fig_deaths.update_layout(
            height=380,
            xaxis_title="Ano",
            yaxis_title="Quantidade",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_deaths, use_container_width=True)

    st.divider()

    # 6. Outros Gráficos (Pie Charts)
    graph_col_1, graph_col_2 = st.columns(2)

    with graph_col_1:
        st.subheader("Peso por gravidade")
        severity = pd.DataFrame(
            {
                "Categoria": ["Feridos leves", "Feridos graves", "Feridos", "Mortos"],
                "Valor": [
                    filtered["feridos_leves"].fillna(0).sum() if "feridos_leves" in filtered.columns else 0,
                    filtered["feridos_graves"].fillna(0).sum() if "feridos_graves" in filtered.columns else 0,
                    filtered["feridos"].fillna(0).sum() if "feridos" in filtered.columns else 0,
                    filtered["mortos"].fillna(0).sum() if "mortos" in filtered.columns else 0,
                ],
            }
        )

        fig_severity = px.pie(
            severity, 
            values="Valor", 
            names="Categoria", 
            hole=0.45,
            color="Categoria",
            color_discrete_map=COLOR_MAP_SEVERIDADE
        )
        fig_severity.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_severity, use_container_width=True)

    with graph_col_2:
        st.subheader("Tipos de pista")
        tipo_pista_summary = (
            filtered.groupby("tipo_pista", dropna=True)
            .agg(acidentes=("id", "count"))
            .reset_index()
            .sort_values("acidentes", ascending=False)
        )

        COLOR_MAP_PISTA = {
            "Simples": "#93C5FD",  # Azul claro
            "Dupla": "#2563EB",    # Azul royal
            "Múltipla": "#1E3A8A"  # Azul marinho
        }

        fig_tipo_pista = px.pie(
            tipo_pista_summary, 
            values="acidentes", 
            names="tipo_pista", 
            hole=0.45,
            color="tipo_pista",
            color_discrete_map=COLOR_MAP_PISTA
        )
        fig_tipo_pista.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_tipo_pista, use_container_width=True)

    st.divider()

    # 7. Causas e Condições Meteorológicas (Antiga implementação de 2 colunas comentada)
    # cause_col, weather_col = st.columns(2)
    # 
    # with cause_col:
    #     st.subheader("Principais causas de acidente")
    #     cause_summary = (
    #         filtered.groupby("causa_acidente", dropna=True)
    #         .agg(acidentes=("id", "count"), mortos=("mortos", "sum"))
    #         .reset_index()
    #         .sort_values(["acidentes", "mortos"], ascending=False)
    #         .head(10)
    #     )
    #     fig_cause = px.bar(
    #         cause_summary.sort_values("acidentes"),
    #         x="acidentes",
    #         y="causa_acidente",
    #         orientation="h",
    #         color="mortos",
    #         color_continuous_scale=["#FEE2E2", "#991B1B"],
    #     )
    #     fig_cause.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Acidentes", yaxis_title="Causa")
    #     st.plotly_chart(fig_cause, use_container_width=True)
    # 
    # with weather_col:
    #     st.subheader("Acidentes por condição meteorológica")
    #     weather_summary = (
    #         filtered.dropna(subset=["condicao_metereologica"])
    #         .assign(condicao_metereologica=lambda frame: frame["condicao_metereologica"].astype("string").str.strip())
    #         .groupby("condicao_metereologica", dropna=True)
    #         .agg(acidentes=("id", "count"), mortos=("mortos", "sum"))
    #         .reset_index()
    #         .sort_values("acidentes", ascending=False)
    #     )
    # 
    #     if weather_summary.empty:
    #         st.info("Não há dados suficientes para montar o comparativo por condição meteorológica.")
    #     else:
    #         fig_weather = px.bar(
    #             weather_summary.sort_values("acidentes"),
    #             x="acidentes",
    #             y="condicao_metereologica",
    #             orientation="h",
    #             color="mortos",
    #             color_continuous_scale=["#FEE2E2", "#991B1B"],
    #         )
    #         fig_weather.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), coloraxis_colorbar=dict(title="Mortos"))
    #         st.plotly_chart(fig_weather, use_container_width=True)

    # Nova implementação em linhas separadas de largura total
    st.subheader("Principais causas de acidente")
    cause_summary = (
        filtered.groupby("causa_acidente", dropna=True)
        .agg(acidentes=("id", "count"), mortos=("mortos", "sum"))
        .reset_index()
        .sort_values(["acidentes", "mortos"], ascending=False)
        .head(10)
    )
    fig_cause = px.bar(
        cause_summary.sort_values("acidentes"),
        x="acidentes",
        y="causa_acidente",
        orientation="h",
        color="mortos",
        color_continuous_scale=["#FEE2E2", "#991B1B"], # Tons quentes de vermelho
    )
    fig_cause.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Acidentes", yaxis_title="Causa")
    st.plotly_chart(fig_cause, use_container_width=True)

    st.divider()

    st.subheader("Acidentes por condição meteorológica")
    weather_summary = (
        filtered.dropna(subset=["condicao_metereologica"])
        .assign(condicao_metereologica=lambda frame: frame["condicao_metereologica"].astype("string").str.strip())
        .groupby("condicao_metereologica", dropna=True)
        .agg(acidentes=("id", "count"), mortos=("mortos", "sum"))
        .reset_index()
        .sort_values("acidentes", ascending=False)
    )

    if weather_summary.empty:
        st.info("Não há dados suficientes para montar o comparativo por condição meteorológica.")
    else:
        fig_weather = px.bar(
            weather_summary.sort_values("acidentes"),
            x="acidentes",
            y="condicao_metereologica",
            orientation="h",
            color="mortos",
            color_continuous_scale=["#FEE2E2", "#991B1B"], # Cores quentes baseadas em mortes (casos graves)
        )
        fig_weather.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), coloraxis_colorbar=dict(title="Mortos"))
        st.plotly_chart(fig_weather, use_container_width=True)

    st.divider()

    # 8. Heatmap de fase do dia e dia da semana
    st.subheader("Comparativo entre fase do dia e dia da semana")
    fase_dia_base = filtered.dropna(subset=["fase_dia", "dia_semana"]).copy()
    fase_dia_base["fase_dia"] = fase_dia_base["fase_dia"].astype("string").str.strip()
    fase_dia_base["dia_semana"] = fase_dia_base["dia_semana"].astype("string").str.strip()

    fase_dia_order = ["Amanhecer", "Pleno dia", "Anoitecer", "Plena Noite"]
    dia_semana_order = [
        "domingo",
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
    ]

    fase_dia_summary = (
        fase_dia_base.pivot_table(
            index="fase_dia",
            columns="dia_semana",
            values="id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=fase_dia_order, columns=dia_semana_order)
        .fillna(0)
    )

    if fase_dia_summary.empty or fase_dia_summary.to_numpy().sum() == 0:
        st.info("Não há dados suficientes de fase do dia para montar o comparativo.")
    else:
        col_totals = fase_dia_summary.sum(axis=0)
        z_data = fase_dia_summary.to_numpy()
        cols = list(fase_dia_summary.columns)
        rows = list(fase_dia_summary.index)
        text_matrix = []
        for i, row in enumerate(rows):
            row_text = []
            for j, col in enumerate(cols):
                val = z_data[i, j]
                total = col_totals[col]
                pct = (val / total * 100) if total else 0.0
                row_text.append(f"{val}<br>({pct:.1f}%)")
            text_matrix.append(row_text)

        fig_fase_dia = go.Figure(
            data=go.Heatmap(
                x=cols,
                y=rows,
                z=z_data,
                colorscale="YlOrRd",
                text=text_matrix,
                texttemplate="%{text}",
                hovertemplate="Fase do dia=%{y}<br>Dia da semana=%{x}<br>Acidentes=%{z}<extra></extra>",
                colorbar=dict(title="Acidentes"),
            )
        )
        fig_fase_dia.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Dia da semana",
            yaxis_title="Fase do dia",
        )
        fig_fase_dia.update_yaxes(autorange="reversed")  # Reverte a ordem do Plotly para Amanhecer ficar no topo
        st.plotly_chart(fig_fase_dia, use_container_width=True)

    st.divider()

    # 9. Panorama mock
    st.subheader("Acidentes ocorridos em locais sinalizados")
    st.caption("Gráfico independente dos filtros, com dados até 2023.")

    mock_summary = build_plate_match_summary(DATA_URL, PLACAS_DATA_FILE)

    if mock_summary.empty:
        st.info("Não foi possível montar o panorama mockado com os arquivos disponíveis.")
    else:
        fig_mock = go.Figure()
        fig_mock.add_trace(
            go.Bar(
                x=mock_summary["Ano"],
                y=mock_summary["Com placa"],
                name="Com placa",
                marker_color=COLOR_ACIDENTES,
            )
        )
        fig_mock.add_trace(
            go.Bar(
                x=mock_summary["Ano"],
                y=mock_summary["Sem placa"],
                name="Sem placa",
                marker_color=COLOR_MORTOS,
            )
        )
        fig_mock.update_layout(
            barmode="group",
            height=430,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Ano",
            yaxis_title="Quantidade de acidentes",
            legend_title_text="Resultado",
        )
        st.plotly_chart(fig_mock, use_container_width=True)

    st.divider()

    # 10. Tabela comparativa e download
    st.subheader("Tabela comparativa por ano")
    table = year_summary.copy()
    table["mortes_por_1000_acidentes"] = table["mortes_por_1000_acidentes"].round(2)
    table["variacao_acidentes_pct"] = table["variacao_acidentes_pct"].round(2)
    table["variacao_mortos_pct"] = table["variacao_mortos_pct"].round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_export = filtered.to_csv(index=False, sep=";")
    st.download_button(
        label="Baixar base filtrada",
        data=csv_export,
        file_name="datatran_filtrado.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()