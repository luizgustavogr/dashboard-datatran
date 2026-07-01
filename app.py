from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "datatran_unificado.csv"


st.set_page_config(
    page_title="Dashboard DATATRAN",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";", encoding="utf-8", low_memory=False)
    df.columns = [column.strip().lower() for column in df.columns]

    if "data_inversa" not in df.columns:
        raise ValueError("A coluna data_inversa nao foi encontrada no CSV unificado.")

    df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    df["ano"] = df["data_inversa"].dt.year
    df["mes"] = df["data_inversa"].dt.month

    text_columns = ["uf", "municipio", "causa_acidente", "tipo_acidente", "classificacao_acidente", "fase_dia", "tipo_pista"]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

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


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    years = sorted([int(year) for year in df["ano"].dropna().unique()])
    year_min, year_max = st.sidebar.slider("Faixa de anos", min_value=min(years), max_value=max(years), value=(min(years), max(years)))

    filtered = df[df["ano"].between(year_min, year_max)]

    if "uf" in filtered.columns:
        ufs = sorted(filtered["uf"].dropna().astype(str).unique().tolist())
        selected_ufs = st.sidebar.multiselect("UF", ufs, default=ufs)
        if selected_ufs:
            filtered = filtered[filtered["uf"].isin(selected_ufs)]

    if "classificacao_acidente" in filtered.columns:
        classificacoes = sorted(filtered["classificacao_acidente"].dropna().astype(str).unique().tolist())
        selected_classificacoes = st.sidebar.multiselect("Classificacao", classificacoes, default=classificacoes)
        if selected_classificacoes:
            filtered = filtered[filtered["classificacao_acidente"].isin(selected_classificacoes)]

    if "tipo_pista" in filtered.columns:
        tipos_pista = sorted(filtered["tipo_pista"].dropna().astype(str).unique().tolist())
        selected_tipos_pista = st.sidebar.multiselect("Tipo de pista", tipos_pista, default=tipos_pista)
        if selected_tipos_pista:
            filtered = filtered[filtered["tipo_pista"].isin(selected_tipos_pista)]

    return filtered


def add_metric_card(label: str, value: float, delta: str | None = None) -> None:
    if delta is None:
        st.metric(label, format_number(value))
    else:
        st.metric(label, format_number(value), delta)


def main() -> None:
    st.title("Dashboard DATATRAN")
    st.caption("Analise do CSV unificado com comparativos entre anos, estados, causas e gravidade dos acidentes.")

    if not DATA_FILE.exists():
        st.error(f"Arquivo nao encontrado: {DATA_FILE.name}")
        st.stop()

    df = load_data(DATA_FILE)
    df = df.dropna(subset=["ano"])

    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("Nenhum registro corresponde aos filtros selecionados.")
        st.stop()

    year_summary = build_year_summary(filtered)
    latest_year = int(year_summary["ano"].max())
    latest_row = year_summary.loc[year_summary["ano"] == latest_year].iloc[0]

    previous_year = None
    previous_row = None
    if len(year_summary) > 1:
        previous_year = int(year_summary.iloc[-2]["ano"])
        previous_row = year_summary.iloc[-2]

    st.subheader("Indicadores gerais")
    metric_cols = st.columns(5)

    with metric_cols[0]:
        add_metric_card("Acidentes", filtered.shape[0])
    with metric_cols[1]:
        add_metric_card("Mortos", filtered["mortos"].fillna(0).sum())
    with metric_cols[2]:
        add_metric_card("Feridos totais", year_summary["feridos_totais"].sum())
    with metric_cols[3]:
        add_metric_card("Veiculos", filtered["veiculos"].fillna(0).sum())
    with metric_cols[4]:
        delta_text = None
        if previous_row is not None and latest_row["acidentes"] is not None and previous_row["acidentes"]:
            delta_value = ((latest_row["acidentes"] - previous_row["acidentes"]) / previous_row["acidentes"]) * 100
            delta_text = f"{delta_value:.1f}% vs {previous_year}"
        add_metric_card(f"Acidentes em {latest_year}", latest_row["acidentes"], delta_text)

    st.divider()

    left_col, right_col = st.columns((2, 1))

    with left_col:
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
        st.plotly_chart(fig_years, use_container_width=True)

    with right_col:
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
        st.plotly_chart(fig_severity, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 estados")
        uf_summary = (
            filtered.groupby("uf", dropna=True)
            .agg(acidentes=("id", "count"), mortos=("mortos", "sum"))
            .reset_index()
            .sort_values(["acidentes", "mortos"], ascending=False)
            .head(10)
        )
        fig_uf = px.bar(uf_summary, x="uf", y="acidentes", color="mortos", color_continuous_scale="Reds")
        fig_uf.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="UF", yaxis_title="Acidentes")
        st.plotly_chart(fig_uf, use_container_width=True)

    with col2:
        st.subheader("Top 10 causas")
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
        st.plotly_chart(fig_cause, use_container_width=True)

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