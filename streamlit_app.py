import streamlit as st
import pandas as pd
import lasio
import io
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re # ¡Asegúrate de que 're' esté importado!

# --- Función de utilidad: Extract_speed_from_filename (MOVIDA AQUÍ) ---
def extract_speed_from_filename(filename):
    """Extracts the speed (e.g., 30, 60, 90, 120) from the filename."""
    match = re.search(r'_(\d+)-FPM_', filename)
    if match:
        return int(match.group(1))
    return None
# --- FIN DE LA FUNCIÓN DE UTILIDAD ---


st.set_page_config(layout="wide")
st.title('Análisis y Visualización de Registros PLT (Formato LAS)')

# Sección de Carga de Múltiples Archivos
uploaded_files = st.file_uploader("Sube tus archivos de registro PLT (.LAS)", type=["las"], accept_multiple_files=True)
all_dfs = {}

if uploaded_files:
    st.subheader("Archivos Cargados y Procesados:")
    for uploaded_file_item in uploaded_files:
        file_name = uploaded_file_item.name
        st.write(f"Procesando: **{file_name}**")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
            tmp_file.write(uploaded_file_item.getvalue())
            tmp_file_path = tmp_file.name
        try:
            las_file = lasio.read(tmp_file_path, encoding='latin-1') # O 'cp1252'
            df = las_file.df()
            all_dfs[file_name] = df
            st.success(f"'{file_name}' cargado y procesado exitosamente.")
        except Exception as e:
            st.error(f"Error al cargar o procesar '{file_name}': {e}")
            st.info(f"Asegúrate de que '{file_name}' sea un formato .LAS válido y no esté corrupto.")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    # --- Clasificación de Pasadas ---
    if all_dfs:
        st.write("---")
        st.subheader("Clasificación de Pasadas:")
        PASADAS_SUBIENDO = {}
        PASADAS_BAJANDO = {}
        for file_name, df_content in all_dfs.items():
            if "SUBIENDO" in file_name.upper():
                PASADAS_SUBIENDO[file_name] = df_content
            elif "BAJANDO" in file_name.upper():
                PASADAS_BAJANDO[file_name] = df_content

        st.write("Archivos clasificados como 'SUBIENDO':")
        if PASADAS_SUBIENDO:
            for file_name in PASADAS_SUBIENDO.keys():
                st.write(f"- {file_name}")
        else:
            st.write("Ningún archivo clasificado como 'SUBIENDO'.")

        st.write("\nArchivos clasificados como 'BAJANDO':")
        if PASADAS_BAJANDO:
            for file_name in PASADAS_BAJANDO.keys():
                st.write(f"- {file_name}")
        else:
            st.write("Ningún archivo clasificado como 'BAJANDO'.")

        # --- CONSOLIDACIÓN DE DATAFRAMES (Mostrando solo df.head()) ---
        st.write("---")
        st.subheader("Resultados de la Consolidación:")

        df_subiendo = pd.DataFrame()
        if PASADAS_SUBIENDO:
            subiendo_dfs_list = []
            for file_name, df_sub in PASADAS_SUBIENDO.items():
                df_sub_copy = df_sub.copy()
                df_sub_copy['Pass'] = file_name
                subiendo_dfs_list.append(df_sub_copy)
            df_subiendo = pd.concat(subiendo_dfs_list)
            st.success("DataFrame 'subiendo' consolidado exitosamente.")
            st.write("Combined 'subiendo' DataFrame head:")
            st.dataframe(df_subiendo.head())
        else:
            st.info("No hay archivos 'SUBIENDO' para consolidar.")
            df_subiendo = pd.DataFrame()

        df_bajando = pd.DataFrame()
        if PASADAS_BAJANDO:
            bajando_dfs_list = []
            for file_name, df_baj in PASADAS_BAJANDO.items():
                df_baj_copy = df_baj.copy()
                df_baj_copy['Pass'] = file_name
                bajando_dfs_list.append(df_baj_copy)
            df_bajando = pd.concat(bajando_dfs_list)
            st.success("DataFrame 'bajando' consolidado exitosamente.")
            st.write("Combined 'bajando' DataFrame head:")
            st.dataframe(df_bajando.head())
        else:
            st.info("No hay archivos 'BAJANDO' para consolidar.")
            df_bajando = pd.DataFrame()

        # --- Control de Calidad (Imputación de Nulos) ---
        st.write("---")
        st.subheader("Control de Calidad: Imputación de Valores Nulos")
        if not df_subiendo.empty:
            st.write("#### Procesando 'df_subiendo'...")
            for col in df_subiendo.select_dtypes(include=['number']).columns:
                if df_subiendo[col].isnull().any():
                    median_value = df_subiendo[col].median()
                    df_subiendo[col].fillna(median_value, inplace=True)
            final_nulls_subiendo = df_subiendo.isnull().sum()
            st.success("Valores nulos en 'df_subiendo' imputados con la mediana.")
            st.write("Valores nulos en 'df_subiendo' después de la imputación:")
            st.dataframe(final_nulls_subiendo[final_nulls_subiendo > 0])
            if final_nulls_subiendo.sum() == 0:
                st.info("¡Todas las columnas numéricas en 'df_subiendo' han sido limpiadas de valores nulos!")
            else:
                st.warning("Algunas columnas no numéricas aún pueden contener valores nulos en 'df_subiendo'.")
        if not df_bajando.empty:
            st.write("#### Procesando 'df_bajando'...")
            for col in df_bajando.select_dtypes(include=['number']).columns:
                if df_bajando[col].isnull().any():
                    median_value = df_bajando[col].median()
                    df_bajando[col].fillna(median_value, inplace=True)
            final_nulls_bajando = df_bajando.isnull().sum()
            st.success("Valores nulos en 'df_bajando' imputados con la mediana.")
            st.write("Valores nulos en 'df_bajando' después de la imputación:")
            st.dataframe(final_nulls_bajando[final_nulls_bajando > 0])
            if final_nulls_bajando.sum() == 0:
                st.info("¡Todas las columnas numéricas en 'df_bajando' han sido limpiadas de valores nulos!")
            else:
                st.warning("Algunas columnas no numéricas aún pueden contener valores nulos en 'df_bajando'.")
        if df_subiendo.empty and df_bajando.empty:
            st.info("No hay DataFrames para procesar valores nulos.")

        # --- GRÁFICAS INTERACTIVAS CON STREAMLIT ---

        st.write("---")
        st.subheader("Visualización Interactiva de Registros PLT")

        if df_subiendo.empty and df_bajando.empty:
            st.warning("No hay datos cargados para generar gráficas.")
        else:
            # 1. Añadir la columna 'Speed' si no existe o es toda nula (esto se hace aquí)
            # Aunque ya la agregaste en tu Colab, nos aseguramos que esté antes de intentar usarla.
            if not df_subiendo.empty and ('Speed' not in df_subiendo.columns or df_subiendo['Speed'].isnull().all()):
                df_subiendo['Speed'] = df_subiendo['Pass'].apply(extract_speed_from_filename)
            if not df_bajando.empty and ('Speed' not in df_bajando.columns or df_bajando['Speed'].isnull().all()):
                df_bajando['Speed'] = df_bajando['Pass'].apply(extract_speed_from_filename)

            # 2. Definir las columnas disponibles para graficar
            all_plot_columns = []
            if not df_subiendo.empty:
                all_plot_columns = [col for col in df_subiendo.columns.tolist() if col not in ['Pass', 'Speed', 'DEPT']]
            elif not df_bajando.empty:
                 all_plot_columns = [col for col in df_bajando.columns.tolist() if col not in ['Pass', 'Speed', 'DEPT']]

            if not all_plot_columns:
                st.warning("No se encontraron columnas adecuadas para graficar (excluyendo 'Pass', 'Speed', 'DEPT').")
            else:
                # 3. Widgets de Streamlit para selección de columnas y velocidad
                st.sidebar.header("Opciones de Gráfica")
                selected_columns = st.sidebar.multiselect(
                    'Selecciona Columnas para Graficar:',
                    options=all_plot_columns,
                    default=all_plot_columns[:min(4, len(all_plot_columns))]
                )

                # Extraer velocidades únicas de ambos DataFrames
                all_speeds = []
                if not df_subiendo.empty and 'Speed' in df_subiendo.columns:
                    speeds_subiendo = df_subiendo['Speed'].dropna().unique().tolist()
                    all_speeds.extend(speeds_subiendo)
                if not df_bajando.empty and 'Speed' in df_bajando.columns:
                    speeds_bajando = df_bajando['Speed'].dropna().unique().tolist()
                    all_speeds.extend(speeds_bajando)

                all_speeds = sorted(list(set(all_speeds))) # Eliminar duplicados y ordenar

                speed_options = [('Todas las Velocidades', None)] + [(str(speed), speed) for speed in all_speeds]

                selected_speed_subiendo = st.sidebar.selectbox(
                    'Selecciona Velocidad (Subiendo):',
                    options=[option[0] for option in speed_options],
                    format_func=lambda x: x
                )
                selected_speed_subiendo_value = next((option[1] for option in speed_options if option[0] == selected_speed_subiendo), None)


                selected_speed_bajando = st.sidebar.selectbox(
                    'Selecciona Velocidad (Bajando):',
                    options=[option[0] for option in speed_options],
                    format_func=lambda x: x
                )
                selected_speed_bajando_value = next((option[1] for option in speed_options if option[0] == selected_speed_bajando), None)


                # Rango de profundidad
                min_overall_dept = None
                max_overall_dept = None
                if not df_subiendo.empty:
                    min_overall_dept = df_subiendo.index.min()
                    max_overall_dept = df_subiendo.index.max()
                if not df_bajando.empty:
                    if min_overall_dept is None or df_bajando.index.min() < min_overall_dept:
                        min_overall_dept = df_bajando.index.min()
                    if max_overall_dept is None or df_bajando.index.max() > max_overall_dept:
                        max_overall_dept = df_bajando.index.max()

                if min_overall_dept is not None and max_overall_dept is not None:
                    min_dept = st.sidebar.slider(
                        'Profundidad Mínima:',
                        min_value=float(min_overall_dept),
                        max_value=float(max_overall_dept),
                        value=float(min_overall_dept),
                        step=0.5
                    )
                    max_dept = st.sidebar.slider(
                        'Profundidad Máxima:',
                        min_value=float(min_overall_dept),
                        max_value=float(max_overall_dept),
                        value=float(max_overall_dept),
                        step=0.5
                    )
                else:
                    st.sidebar.warning("No hay datos para establecer el rango de profundidad.")
                    min_dept = None
                    max_dept = None


                # Función de ploteo para Streamlit
                def plot_qc_logs_streamlit(df, title_prefix, selected_columns, selected_speed=None, min_dept=None, max_dept=None):
                    if not selected_columns:
                        st.warning(f"No hay columnas seleccionadas para graficar en {title_prefix}.")
                        return

                    df_filtered = df.copy()
                    if selected_speed is not None:
                        df_filtered = df_filtered[df_filtered['Speed'] == selected_speed]

                    if min_dept is not None and max_dept is not None:
                        df_filtered = df_filtered[(df_filtered.index >= min_dept) & (df_filtered.index <= max_dept)]

                    unique_passes = df_filtered['Pass'].unique()
                    num_cols = len(selected_columns)

                    if num_cols == 0:
                        st.warning(f"No hay columnas válidas seleccionadas para graficar en {title_prefix}.")
                        return

                    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(4 * num_cols, 15), sharey=True)

                    if num_cols == 1:
                        axes = [axes]

                    for i, col in enumerate(selected_columns):
                        ax = axes[i]
                        for pass_name in unique_passes:
                            df_pass = df_filtered[df_filtered['Pass'] == pass_name]
                            if not df_pass.empty and col in df_pass.columns:
                                ax.plot(df_pass[col], df_pass.index, label=pass_name)
                            else:
                                if not df_pass.empty: # Only warn if DataFrame is not empty but column is missing
                                    st.warning(f"Columna '{col}' no encontrada o vacía en la pasada '{pass_name}' para {title_prefix}.")

                        ax.set_title(f'{col} Track - {title_prefix}')
                        ax.set_xlabel(col)
                        ax.grid(True)

                    axes[0].set_ylabel('DEPT')

                    if not df_filtered.empty:
                        axes[0].set_ylim(df_filtered.index.max(), df_filtered.index.min())
                    else:
                        st.warning(f"DataFrame filtrado para {title_prefix} está vacío. No se puede establecer el rango de profundidad.")

                    if unique_passes.size > 0:
                        axes[0].legend(title="Pass", loc='upper right')
                    else:
                        st.info(f"No hay pasadas para mostrar en las gráficas de {title_prefix} con los filtros seleccionados.")

                    plt.suptitle(f'{title_prefix} Pass Quality Control Plots', y=1.02, fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

                    st.pyplot(fig)
                    plt.close(fig)


                if not df_subiendo.empty and selected_columns:
                    st.write("### Gráficas de Control de Calidad: Pasadas SUBIENDO")
                    plot_qc_logs_streamlit(df_subiendo, 'Subiendo', selected_columns,
                                           selected_speed=selected_speed_subiendo_value,
                                           min_dept=min_dept, max_dept=max_dept)
                else:
                    st.info("No hay DataFrames 'SUBIENDO' o columnas seleccionadas para graficar.")

                if not df_bajando.empty and selected_columns:
                    st.write("### Gráficas de Control de Calidad: Pasadas BAJANDO")
                    plot_qc_logs_streamlit(df_bajando, 'Bajando', selected_columns,
                                           selected_speed=selected_speed_bajando_value,
                                           min_dept=min_dept, max_dept=max_dept)
                else:
                    st.info("No hay DataFrames 'BAJANDO' o columnas seleccionadas para graficar.")
    else:
        st.error("No se pudieron cargar archivos LAS. Por favor, revisa tus archivos.")

else:
    st.info("Por favor, sube uno o más archivos .LAS para comenzar el análisis.")