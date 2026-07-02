from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://github.com/luizgustavogr/dashboard-datatran/releases/download/v1.0/datatran_unificado.csv"
LOCAL_DATA_FILE = BASE_DIR / "datatran_unificado.csv"
PLACAS_DATA_FILE = BASE_DIR / "placas_sinalizacao_processado.csv"
RADARES_DATA_FILE = BASE_DIR / "radares_velocidade_processado.csv"
MOCK_END_YEAR = 2023
USECOLS = [
    "id",
    "data_inversa",
    "dia_semana",
    "br",
    "km",
    "uf",
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

    text_columns = ["uf", "causa_acidente", "tipo_acidente", "classificacao_acidente", "condicao_metereologica", "fase_dia", "tipo_pista", "dia_semana"]
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

    summary["mortes_por_1000_acidentes"] = summary["mortos"] / summary["acidentes"].replace(0, pd.NA) * 1000
    summary["feridos_totais"] = summary[["feridos_leves", "feridos_graves", "feridos"]].fillna(0).sum(axis=1)
    summary["variacao_acidentes_pct"] = summary["acidentes"].pct_change() * 100
    summary["variacao_mortos_pct"] = summary["mortos"].pct_change() * 100
    return summary


@st.cache_data(show_spinner=False)
def build_mock_year_summary(
    source: str | Path,
    series_name: str,
    year_column: str | None = None,
    date_column: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(source, sep=";", low_memory=False, encoding="utf-8-sig")
    df.columns = [column.strip().lower() for column in df.columns]

    if year_column and year_column in df.columns:
        df["ano"] = pd.to_numeric(df[year_column], errors="coerce")
    elif date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df["ano"] = df[date_column].dt.year
    else:
        raise ValueError(f"Nao foi possivel identificar o ano no arquivo {source}.")

    summary = (
        df.dropna(subset=["ano"])
        .assign(ano=lambda frame: frame["ano"].astype(int))
        .groupby("ano", dropna=True)
        .size()
        .reset_index(name=series_name)
    )

    return summary[summary["ano"] <= MOCK_END_YEAR]


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


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filtros")

    filter_row_1 = st.columns(3)

    years = sorted([int(year) for year in df["ano"].dropna().unique()])
    with filter_row_1[0]:
        selected_years = st.multiselect("Faixa de anos", years, default=years)

    if selected_years:
        filtered = df[df["ano"].isin(selected_years)]
    else:
        filtered = df.iloc[0:0]

    if "uf" in filtered.columns:
        ufs = sorted(filtered["uf"].dropna().astype(str).unique().tolist())
        with filter_row_1[1]:
            selected_ufs = st.multiselect("UF", ufs, default=ufs)
        if selected_ufs:
            filtered = filtered[filtered["uf"].isin(selected_ufs)]

    if "classificacao_acidente" in filtered.columns:
        classificacoes = sorted(filtered["classificacao_acidente"].dropna().astype(str).unique().tolist())
        with filter_row_1[2]:
            selected_classificacoes = st.multiselect("Classificacao", classificacoes, default=classificacoes)
        if selected_classificacoes:
            filtered = filtered[filtered["classificacao_acidente"].isin(selected_classificacoes)]

    return filtered


def add_metric_card(label: str, value: float, delta: str | None = None) -> None:
    if delta is None:
        st.metric(label, format_number(value))
    else:
        st.metric(label, format_number(value), delta)


def main() -> None:
    st.title("Dashboard DATATRAN")
    st.caption("Análise de dados sobre acidentes em rodovias federais brasileiras.")

    try:
        df = load_data(DATA_URL)
    except Exception:
        st.error("Nao foi possivel carregar os dados.")
        st.stop()

    df = df.dropna(subset=["ano"])

    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("Nenhum registro corresponde aos filtros selecionados.")
        st.stop()

    year_summary = build_year_summary(filtered)

    st.subheader("Indicadores gerais")
    metric_cols = st.columns(4)

    with metric_cols[0]:
        add_metric_card("Acidentes", filtered.shape[0])
    with metric_cols[1]:
        add_metric_card("Mortos", filtered["mortos"].fillna(0).sum())
    with metric_cols[2]:
        add_metric_card("Feridos totais", year_summary["feridos_totais"].sum())
    with metric_cols[3]:
        add_metric_card("Veiculos", filtered["veiculos"].fillna(0).sum())

    st.divider()

    year_left_col, year_center_col, year_right_col = st.columns((1, 2, 1))

    with year_center_col:
        st.subheader("Comparativo entre anos")
        fig_years = go.Figure()
        fig_years.add_trace(
            go.Bar(
                x=year_summary["ano"],
                y=year_summary["acidentes"],
                name="Acidentes",
                marker_color="#2E86DE",
            )
        )
        fig_years.add_trace(
            go.Scatter(
                x=year_summary["ano"],
                y=year_summary["mortos"],
                name="Mortos",
                mode="lines+markers",
                line=dict(color="#C0392B", width=3),
            )
        )
        fig_years.update_layout(
            barmode="group",
            height=450,
            xaxis_title="Ano",
            yaxis_title="Quantidade",
            legend_title_text="Serie",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_years, width="stretch")

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

        fig_severity = px.pie(severity, values="Valor", names="Categoria", hole=0.45)
        fig_severity.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_severity, width="stretch")

    with graph_col_2:
        st.subheader("Tipos de pista")
        tipo_pista_summary = (
            filtered.groupby("tipo_pista", dropna=True)
            .agg(acidentes=("id", "count"))
            .reset_index()
            .sort_values("acidentes", ascending=False)
        )

        fig_tipo_pista = px.pie(tipo_pista_summary, values="acidentes", names="tipo_pista", hole=0.45)
        fig_tipo_pista.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_tipo_pista, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comparativo entre os principais estados")
        uf_summary = (
            filtered.groupby("uf", dropna=True)
            .agg(acidentes=("id", "count"), mortos=("mortos", "sum"))
            .reset_index()
            .sort_values(["acidentes", "mortos"], ascending=False)
            .head(10)
        )
        fig_uf = px.bar(uf_summary, x="uf", y="acidentes", color="mortos", color_continuous_scale="Reds")
        fig_uf.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="UF", yaxis_title="Acidentes")
        st.plotly_chart(fig_uf, width="stretch")

    with col2:
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
            color_continuous_scale="OrRd",
        )
        fig_cause.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Acidentes", yaxis_title="Causa")
        st.plotly_chart(fig_cause, width="stretch")

    st.subheader("Acidentes por condição meteorológica")
    weather_summary = (
        filtered.dropna(subset=["condicao_metereologica"])
        .assign(condicao_metereologica=lambda frame: frame["condicao_metereologica"].astype("string").str.strip())
        .groupby("condicao_metereologica", dropna=True)
        .agg(acidentes=("id", "count"))
        .reset_index()
        .sort_values("acidentes", ascending=False)
    )

    if weather_summary.empty:
        st.info("Nao ha dados suficientes para montar o comparativo por condicao meteorologica.")
    else:
        fig_weather = px.bar(
            weather_summary.sort_values("acidentes"),
            x="acidentes",
            y="condicao_metereologica",
            orientation="h",
            color="acidentes",
            color_continuous_scale="Blues",
        )
        fig_weather.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Acidentes",
            yaxis_title="Condicao meteorologica",
        )
        st.plotly_chart(fig_weather, width="stretch")

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
        st.info("Nao ha dados suficientes de fase do dia para montar o comparativo.")
    else:
        fig_fase_dia = go.Figure(
            data=go.Heatmap(
                x=list(fase_dia_summary.columns),
                y=list(fase_dia_summary.index),
                z=fase_dia_summary.to_numpy(),
                colorscale="YlOrRd",
                text=fase_dia_summary.to_numpy(),
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
        st.plotly_chart(fig_fase_dia, width="stretch")

    st.subheader("Tabela comparativa por ano")
    table = year_summary.copy()
    table["mortes_por_1000_acidentes"] = table["mortes_por_1000_acidentes"].round(2)
    table["variacao_acidentes_pct"] = table["variacao_acidentes_pct"].round(2)
    table["variacao_mortos_pct"] = table["variacao_mortos_pct"].round(2)
    st.dataframe(table, width="stretch", hide_index=True)

    csv_export = filtered.to_csv(index=False, sep=";")
    st.download_button(
        label="Baixar base filtrada",
        data=csv_export,
        file_name="datatran_filtrado.csv",
        mime="text/csv",
    )

    st.subheader("Acidentes ocorridos em locais sinalizados")
    st.caption("Gráfico independente dos filtros, com dados até 2023.")

    mock_summary = build_plate_match_summary(DATA_URL, PLACAS_DATA_FILE)

    if mock_summary.empty:
        st.info("Nao foi possivel montar o panorama mockado com os arquivos disponiveis.")
    else:
        fig_mock = go.Figure()
        fig_mock.add_trace(
            go.Bar(
                x=mock_summary["Ano"],
                y=mock_summary["Com placa"],
                name="Com placa",
                marker_color="#2E86DE",
            )
        )
        fig_mock.add_trace(
            go.Bar(
                x=mock_summary["Ano"],
                y=mock_summary["Sem placa"],
                name="Sem placa",
                marker_color="#C0392B",
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
        st.plotly_chart(fig_mock, width="stretch")


if __name__ == "__main__":
    main()