# app.py
# Este script implementa una aplicación de Streamlit para el análisis y
# visualización de registros PLT y PBU en formato LAS.
# El código ha sido limpiado de las funcionalidades de autenticación.

import streamlit as st
import pandas as pd
import lasio
import io
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# --- Configuración de la página ---
# Se establece la configuración de la página, incluyendo el diseño amplio.
st.set_page_config(layout="wide")

# Se carga y muestra la imagen del logo.
st.image("images/1.-Ecopetrol.png", width=200)

# Título y descripción de la aplicación.
st.title('Análisis y Visualización de Registros PLT y PBU (Formato LAS)')
st.write("---")

# --- Mapeo de columnas por compañía para PBU ---
# Este diccionario define la equivalencia de nombres de columnas para diferentes compañías
# y los mapea a un conjunto de nombres estándar que el resto del script utilizará.
COMPANY_COL_MAPPING = {
    "EXPRO": {
        'ETIM': ['ELTIM'],         # Tiempo transcurrido
        'WPRE': ['QPS_PRES'],      # Presión de pozo
        'DEPTH': ['ADPTH'],        # Profundidad del medidor
        'GR': ['GTC_GR'],          # Gamma Ray
        'WTEP': ['GTC_WTemp'],     # Temperatura del pozo
        'SPIN:1': ['IFS'],         # Velocidad de spinner 1
        'SPIN:2': ['CFM'],         # Velocidad de spinner 2
        'WFDE': ['QCD_FDEN'],      # Densidad de fluido
    },
    "SLB": {
        'ETIM': ['ETIM'],          # Tiempo transcurrido
        'WPRE': ['WPRE'],          # Presión de pozo
        'DEPTH': ['TDEP'],         # Profundidad del medidor
        'GR': ['GR'],              # Gamma Ray
        'WTEP': ['WTEP'],          # Temperatura del pozo
        'SPIN:1': ['SPIN', 'SPIN:1', 'SPIN_1'],  # Adaptación para posibles nombres
        'SPIN:2': ['SPIN:2', 'SPIN_2'],         # Adaptación para posibles nombres
        'WFDE': ['WFDE'],          # Densidad de fluido
    },
    "Desconocido": {} # No se realiza mapeo para compañías no identificadas
}

# --- Funciones de utilidad ---
def extract_speed_from_filename(filename):
    """Extrae la velocidad (ej. 30, 60, 90, 120) del nombre del archivo."""
    match = re.search(r'_(\d+)-FPM_', filename)
    if match:
        return int(match.group(1))
    return None

def detect_compania(las_file_content):
    """
    Lee las primeras líneas del archivo LAS para identificar la compañía.
    """
    encodings_to_try = ['latin-1', 'utf-8', 'cp1252']
    header_content = ""
    
    # Asume que las_file_content es un objeto de archivo en memoria.
    # Lee las primeras 2048 bytes para detectar el encabezado.
    content_to_check = las_file_content.read(2048)
    las_file_content.seek(0) # Vuelve al inicio del archivo
    
    for encoding in encodings_to_try:
        try:
            header_content = content_to_check.decode(encoding, errors='ignore')
            break
        except UnicodeDecodeError:
            continue
    
    if "EXPRO" in header_content.upper():
        return "EXPRO"
    elif "SLB" in header_content.upper() or "SCHLUMBERGER" in header_content.upper():
        return "SLB"
    else:
        return "Desconocido"

def map_columns_to_standard(df, company_detected):
    """
    Mapea y renombra las columnas de un DataFrame a nombres estándar
    basados en la compañía detectada.
    """
    mapping = COMPANY_COL_MAPPING.get(company_detected, {})
    renamed_cols = {}
    for standard_name, possible_names in mapping.items():
        for name in possible_names:
            if name in df.columns:
                renamed_cols[name] = standard_name
                break
    
    df.rename(columns=renamed_cols, inplace=True)
    return df

def plot_qc_logs_streamlit(df, title_prefix, selected_columns, formation_intervals_df=None, selected_speed=None, min_dept=None, max_dept=None):
    """Genera y muestra gráficas de control de calidad para el DataFrame dado, con intervalos de formación."""
    if not selected_columns:
        st.warning(f"No hay columnas seleccionadas para graficar en {title_prefix}.")
        return

    df_filtered = df.copy()
    if selected_speed is not None and 'Speed' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Speed'] == selected_speed]

    if min_dept is not None and max_dept is not None and not df_filtered.empty:
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
                if not df_pass.empty:
                    st.warning(f"Columna '{col}' no encontrada o vacía en la pasada '{pass_name}' para {title_prefix}.")

        ax.set_title(f'{col} Track - {title_prefix}')
        ax.set_xlabel(col)
        ax.grid(True)
        
        # --- Dibuja los intervalos de formación ---
        if formation_intervals_df is not None and not formation_intervals_df.empty:
            for _, row in formation_intervals_df.iterrows():
                try:
                    # Convierte los valores a flotantes para asegurar que son numéricos
                    top = float(row['Tope (ft)'])
                    base = float(row['Base (ft)'])
                    formation_name = str(row['Formación'])
                    
                    # Dibuja el área sombreada para el intervalo de la formación
                    ax.axhspan(base, top, color='lightgray', alpha=0.5)
                    
                    # Anota el nombre de la formación en el centro del intervalo
                    text_y = (top + base) / 2
                    ax.text(ax.get_xlim()[0], text_y, formation_name,
                            ha='left', va='center', fontsize=10, rotation=90, color='black',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                except (ValueError, TypeError):
                    st.warning("Verifica que los valores de 'Tope' y 'Base' sean numéricos.")
                    continue


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

def process_and_classify_files_plt(uploaded_files):
    """Procesa los archivos .las y los clasifica en subiendo o bajando."""
    all_dfs = {}
    if not uploaded_files:
        st.info("Por favor, sube uno o más archivos .LAS para comenzar el análisis.")
        return None, None

    for uploaded_file_item in uploaded_files:
        file_name = uploaded_file_item.name
        st.write(f"Procesando: **{file_name}**")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
            tmp_file.write(uploaded_file_item.getvalue())
            tmp_file_path = tmp_file.name
        try:
            las_file = lasio.read(tmp_file_path, encoding='latin-1')
            df = las_file.df()
            all_dfs[file_name] = df
            st.success(f"'{file_name}' cargado y procesado exitosamente.")
        except Exception as e:
            st.error(f"Error al cargar o procesar '{file_name}': {e}")
            st.info(f"Asegúrate de que '{file_name}' sea un formato .LAS válido y no esté corrupto.")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    PASADAS_SUBIENDO = {name: df for name, df in all_dfs.items() if "SUBIENDO" in name.upper()}
    PASADAS_BAJANDO = {name: df for name, df in all_dfs.items() if "BAJANDO" in name.upper()}

    return PASADAS_SUBIENDO, PASADAS_BAJANDO

def consolidate_dataframes_plt(pass_dfs, pass_type):
    """Consolida los DataFrames de las pasadas de un tipo específico."""
    if not pass_dfs:
        st.info(f"No hay archivos '{pass_type}' para consolidar.")
        return pd.DataFrame()
    
    dfs_list = []
    for file_name, df_content in pass_dfs.items():
        df_copy = df_content.copy()
        df_copy['Pass'] = file_name
        df_copy['Speed'] = extract_speed_from_filename(file_name)
        dfs_list.append(df_copy)
    
    df_consolidated = pd.concat(dfs_list)
    st.success(f"DataFrame '{pass_type}' consolidado exitosamente.")
    st.write(f"Combined '{pass_type}' DataFrame head:")
    st.dataframe(df_consolidated.head())
    
    return df_consolidated

def impute_nulls_plt(df, title):
    """Imputa los valores nulos con la mediana para columnas numéricas."""
    if df.empty:
        return df

    st.write(f"#### Procesando '{title}'...")
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
    
    final_nulls = df.isnull().sum()
    st.success(f"Valores nulos en '{title}' imputados con la mediana.")
    st.write(f"Valores nulos en '{title}' después de la imputación:")
    st.dataframe(final_nulls[final_nulls > 0])
    
    if final_nulls.sum() == 0:
        st.info(f"¡Todas las columnas numéricas en '{title}' han sido limpiadas!")
    else:
        st.warning(f"Algunas columnas no numéricas aún pueden contener valores nulos en '{title}'.")
        
    return df


# --- Contenedor de carga de archivos (para la pestaña de Pasadas) ---
uploaded_files = st.sidebar.file_uploader(
    "Sube tus archivos de registro (.LAS) para Análisis de Pasadas", 
    type=["las"], 
    accept_multiple_files=True
)

# --- Contenedor de carga de archivos (para la pestaña de PBU) ---
uploaded_file_pbu = st.sidebar.file_uploader(
    "Sube un archivo de registro (.LAS) para Análisis de PBU",
    type=["las"],
    key="pbu_file_uploader"
)


# --- Definición de pestañas ---
tab1, tab2 = st.tabs(["Análisis de Pasadas PLT", "Análisis de PBU"])

with tab1:
    st.header("Análisis de Pasadas y Control de Calidad")
    
    if uploaded_files:
        # Procesamiento y clasificación
        PASADAS_SUBIENDO, PASADAS_BAJANDO = process_and_classify_files_plt(uploaded_files)

        st.subheader("Clasificación de Pasadas:")
        st.write("Archivos clasificados como 'SUBIENDO':")
        if PASADAS_SUBIENDO:
            for file_name in PASADAS_SUBIENDO.keys():
                st.write(f"- {file_name}")
        else:
            st.write("Ningún archivo clasificado como 'SUBIENDO'.")
        st.write("Archivos clasificados como 'BAJANDO':")
        if PASADAS_BAJANDO:
            for file_name in PASADAS_BAJANDO.keys():
                st.write(f"- {file_name}")
        else:
            st.write("Ningún archivo clasificado como 'BAJANDO'.")
            
        st.write("---")
        st.subheader("Consolidación y Control de Calidad")
        
        # Consolidación
        df_subiendo = consolidate_dataframes_plt(PASADAS_SUBIENDO, 'subiendo')
        df_bajando = consolidate_dataframes_plt(PASADAS_BAJANDO, 'bajando')
        
        # Imputación de nulos
        df_subiendo = impute_nulls_plt(df_subiendo, 'df_subiendo')
        df_bajando = impute_nulls_plt(df_bajando, 'df_bajando')
        
        st.write("---")
        st.subheader("Visualización Interactiva de Registros PLT")
        st.markdown("---")

        # --- Entrada de datos de formaciones ---
        st.subheader("Definir Intervalos de Formación")
        st.markdown("Usa la tabla de abajo para ingresar los nombres de las formaciones y sus profundidades (Tope y Base).")

        # Define la estructura del DataFrame para el editor de datos
        default_formation_data = {
            "Formación": ["", ""],
            "Tope (ft)": [0, 0],
            "Base (ft)": [0, 0]
        }
        formation_df = pd.DataFrame(default_formation_data)
        
        # Permite al usuario editar la tabla
        formation_editor = st.data_editor(formation_df, num_rows="dynamic", hide_index=True)
        
        st.markdown("---")

        if df_subiendo.empty and df_bajando.empty:
            st.warning("No hay datos cargados para generar gráficas.")
        else:
            # Definir las columnas disponibles para graficar
            all_plot_columns = []
            if not df_subiendo.empty:
                all_plot_columns = [col for col in df_subiendo.columns.tolist() if col not in ['Pass', 'Speed', 'DEPT']]
            elif not df_bajando.empty:
                all_plot_columns = [col for col in df_bajando.columns.tolist() if col not in ['Pass', 'Speed', 'DEPT']]

            if not all_plot_columns:
                st.warning("No se encontraron columnas adecuadas para graficar (excluyendo 'Pass', 'Speed', 'DEPT').")
            else:
                # Widgets de Streamlit para selección de columnas y velocidad
                selected_columns = st.multiselect(
                    'Selecciona Columnas para Graficar:',
                    options=all_plot_columns,
                    default=all_plot_columns[:min(4, len(all_plot_columns))]
                )

                # Extraer velocidades únicas de ambos DataFrames
                all_speeds = []
                if not df_subiendo.empty and 'Speed' in df_subiendo.columns:
                    all_speeds.extend(df_subiendo['Speed'].dropna().unique().tolist())
                if not df_bajando.empty and 'Speed' in df_bajando.columns:
                    all_speeds.extend(df_bajando['Speed'].dropna().unique().tolist())
                all_speeds = sorted(list(set(all_speeds)))

                speed_options = [('Todas las Velocidades', None)] + [(str(speed), speed) for speed in all_speeds]

                col1, col2 = st.columns(2)
                with col1:
                    selected_speed_subiendo = st.selectbox(
                        'Selecciona Velocidad (Subiendo):',
                        options=[option[0] for option in speed_options],
                        format_func=lambda x: x
                    )
                    selected_speed_subiendo_value = next((option[1] for option in speed_options if option[0] == selected_speed_subiendo), None)

                with col2:
                    selected_speed_bajando = st.selectbox(
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
                    min_dept, max_dept = st.slider(
                        'Rango de Profundidad:',
                        min_value=float(min_overall_dept),
                        max_value=float(max_overall_dept),
                        value=(float(min_overall_dept), float(max_overall_dept)),
                        step=0.5
                    )
                else:
                    st.warning("No hay datos para establecer el rango de profundidad.")
                    min_dept = None
                    max_dept = None

                # Gráficas
                if not df_subiendo.empty and selected_columns:
                    st.write("### Gráficas de Control de Calidad: Pasadas SUBIENDO")
                    plot_qc_logs_streamlit(
                        df_subiendo, 
                        'Subiendo', 
                        selected_columns,
                        formation_intervals_df=formation_editor,
                        selected_speed=selected_speed_subiendo_value,
                        min_dept=min_dept, 
                        max_dept=max_dept
                    )

                if not df_bajando.empty and selected_columns:
                    st.write("### Gráficas de Control de Calidad: Pasadas BAJANDO")
                    plot_qc_logs_streamlit(
                        df_bajando, 
                        'Bajando', 
                        selected_columns,
                        formation_intervals_df=formation_editor,
                        selected_speed=selected_speed_bajando_value,
                        min_dept=min_dept, 
                        max_dept=max_dept
                    )
    else:
        st.info("Por favor, sube uno o más archivos .LAS en la barra lateral para comenzar el análisis de pasadas.")

with tab2:
    st.header("Análisis de Presión Build-Up (PBU)")
    st.markdown(
        """
        Esta página te permite cargar un único archivo de registro en formato .LAS para
        realizar un análisis de PBU. Ahora soporta archivos de las compañías EXPRO y SLB.
        """
    )
    if uploaded_file_pbu:
        file_name = uploaded_file_pbu.name
        las_data = None
        company = None
        
        # Rewind the file to the beginning to ensure lasio can read it
        uploaded_file_pbu.seek(0)
        
        try:
            # Procesar el archivo subido
            las_data = lasio.read(uploaded_file_pbu, encoding='latin-1')
            st.success(f"'{file_name}' cargado y procesado exitosamente.")
            
            # Rewind the file again for the company detection
            uploaded_file_pbu.seek(0)
            
            # Detectar compañía
            company = detect_compania(uploaded_file_pbu)
            st.info(f"Compañía detectada: **{company}**")

        except Exception as e:
            st.error(f"Error al cargar o procesar el archivo: {e}")
            st.error("Es posible que el archivo esté corrupto o que el formato no sea compatible.")
        
        if las_data and company:
            # Ahora, toda la lógica de procesamiento y graficado de PBU
            # está encapsulada aquí.
            df = las_data.df()
            
            st.write("Vista previa del DataFrame original:")
            st.dataframe(df.head())

            # Mapear las columnas si la compañía es conocida
            df = map_columns_to_standard(df, company)
            if not COMPANY_COL_MAPPING.get(company, {}).items():
                st.info("No se aplicó un mapeo de columnas. Mostrando el DataFrame con columnas originales.")
            else:
                st.write(f"Columnas renombradas para el análisis ({company}):")
                st.dataframe(df.head())
            
            st.write("Columnas disponibles después del mapeo:")
            st.write(list(df.columns))

            # Verificamos que las columnas necesarias existan para los gráficos principales
            if 'ETIM' in df.columns and 'WPRE' in df.columns:
                st.write("---")
                
                # --- PRIMER GRÁFICO: PRESIÓN vs. TIEMPO ---
                st.write("### Gráfico de Restauración de Presión vs. Tiempo")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(df['ETIM'], df['WPRE'], 
                            marker='o', 
                            linestyle='-', 
                            linewidth=0.5,
                            markersize=1,
                            color='blue',
                            label='Restauración de Presión')
                ax1.set_xlabel('Tiempo (ETIM)')
                ax1.set_ylabel('Presión (WPRE)')
                ax1.set_title('Gráfico de Restauración de Presión', fontsize=16, fontweight='bold')
                ax1.legend()
                st.pyplot(fig1)
                plt.close(fig1)

                st.write("---")
                
                # --- SEGUNDO GRÁFICO: GRÁFICO LOG-LOG DE BOURDET (AHORA INTERACTIVO) ---
                st.write("### Gráfico Log-Log Interactivo")
                st.markdown("Ajusta los parámetros para analizar el comportamiento de la presión.")

                pws_delta_t_df = df[['ETIM', 'WPRE']].copy().sort_values(by='ETIM')
                initial_pressure = pws_delta_t_df['WPRE'].dropna().iloc[0] if not pws_delta_t_df['WPRE'].dropna().empty else 0
                pws_delta_t_df['delta_P'] = pws_delta_t_df['WPRE'] - initial_pressure
                pws_delta_t_df['ln_ETIM'] = np.log(pws_delta_t_df['ETIM'] + 1e-9)
                pws_delta_t_df['dPws_dlnETIM'] = pws_delta_t_df['WPRE'].diff() / pws_delta_t_df['ln_ETIM'].diff()
                pws_delta_t_df['bourdet_derivative'] = pws_delta_t_df['dPws_dlnETIM']
                
                plot_df_interactive = pws_delta_t_df[
                    (pws_delta_t_df['ETIM'] > 0) & 
                    (pws_delta_t_df['delta_P'].notna()) & 
                    (pws_delta_t_df['delta_P'] > 0)
                ].copy()
                
                st.sidebar.header("Controles para Gráfico Log-Log")
                min_etim = float(plot_df_interactive['ETIM'].min()) if not plot_df_interactive.empty else 1e-1
                max_etim = float(plot_df_interactive['ETIM'].max()) if not plot_df_interactive.empty else 1e5
                
                x_range = st.sidebar.slider(
                    'Rango de Tiempo (Δt) para Gráfica:',
                    min_value=min_etim,
                    max_value=max_etim,
                    value=(min_etim, max_etim),
                    step=(max_etim - min_etim) / 100
                )
                smoothing_window = st.sidebar.slider(
                    'Ventana de Suavizado para Derivada:',
                    min_value=1,
                    max_value=30,
                    value=5
                )
                
                if smoothing_window == 1:
                    smoothed_derivative = plot_df_interactive['bourdet_derivative']
                else:
                    smoothed_derivative = plot_df_interactive['bourdet_derivative'].rolling(window=smoothing_window, center=True).mean()

                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.loglog(plot_df_interactive['ETIM'], plot_df_interactive['delta_P'], 
                            marker='o', linestyle='', markersize=4, color='blue', label='ΔP')

                plot_df_derivative_smoothed = plot_df_interactive.copy()
                plot_df_derivative_smoothed['bourdet_derivative_smoothed'] = smoothed_derivative
                plot_df_derivative_smoothed = plot_df_derivative_smoothed[
                    (plot_df_derivative_smoothed['ETIM'] > 0) & 
                    (plot_df_derivative_smoothed['bourdet_derivative_smoothed'].notna()) & 
                    (plot_df_derivative_smoothed['bourdet_derivative_smoothed'] > 0)
                ].copy()
                ax2.loglog(plot_df_derivative_smoothed['ETIM'], plot_df_derivative_smoothed['bourdet_derivative_smoothed'], 
                            marker='x', linestyle='--', markersize=4, color='red', label='Bourdet Derivative (Smoothed)')

                ax2.set_xlabel('Tiempo (Δt)')
                ax2.set_ylabel('Presión')
                ax2.set_title('Gráfico Log-Log (Bourdet)', fontsize=16, fontweight='bold')
                ax2.set_xlim(x_range[0], x_range[1])
                ax2.grid(True, which="both", ls="-")
                ax2.legend()
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.error("El archivo no contiene las columnas 'ETIM' o 'WPRE' necesarias para el análisis de PBU.")
    else:
        st.info("Por favor, sube un único archivo .LAS en la barra lateral para el análisis de PBU.")
