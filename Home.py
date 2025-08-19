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
st.set_page_config(layout="wide")
st.image("images/1.-Ecopetrol.png", width=200) 
st.title('Análisis y Visualización de Registros PLT y PBU (Formato LAS)')
st.write("---")

# --- Mapeo de columnas por compañía para PBU ---
# Este diccionario define la equivalencia de nombres de columnas para diferentes compañías
# Se incluyen los nombres hegemónicos de ambas compañías (SLB y EXPRO)
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
        'SPIN:2': ['SPIN:2', 'SPIN_2'],          # Adaptación para posibles nombres
        'WFDE': ['WFDE'],          # Densidad de fluido
    },
    "Desconocido": {} # No se realiza mapeo para compañías no identificadas
}

# --- Funciones de utilidad ---
def extract_speed_from_filename(filename):
    """
    Extrae la velocidad (ej. 30, 60, 90, 120) del nombre del archivo.
    Se ha mejorado la expresión regular para ser más flexible.
    """
    # Buscamos un número seguido de 'FPM' o 'fpm', con guiones o underscores a su alrededor.
    match = re.search(r'[\-_](\d+)[-_]?FPM', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def detect_compania(las_file_path):
    """
    Lee las primeras líneas del archivo LAS para identificar la compañía.
    """
    with open(las_file_path, 'rb') as f:
        las_file_content = f.read(2048) # Leer solo los primeros 2 KB para eficiencia
    
    encodings_to_try = ['latin-1', 'utf-8', 'cp1252']
    header_content = ""
    for encoding in encodings_to_try:
        try:
            header_content = las_file_content.decode(encoding, errors='ignore')
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

def plot_qc_logs_streamlit(df, title_prefix, selected_columns, selected_speed=None, min_dept=None, max_dept=None):
    """Genera y muestra gráficas de control de calidad para el DataFrame dado."""
    if not selected_columns:
        st.warning(f"No hay columnas seleccionadas para graficar en {title_prefix}.")
        return

    df_filtered = df.copy()
    
    # Si se selecciona una velocidad, filtrar el DataFrame
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
    """Procesa los archivos .las guardándolos temporalmente y los clasifica."""
    all_dfs = {}
    if not uploaded_files:
        return None, None

    for uploaded_file_item in uploaded_files:
        file_name = uploaded_file_item.name
        tmp_file_path = None
        try:
            # GUARDAR ARCHIVO TEMPORALMENTE
            file_bytes = uploaded_file_item.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            # Detectar la compañía desde el archivo temporal
            company = detect_compania(tmp_file_path)
            st.info(f"Compañía detectada para '{file_name}': **{company}**")

            # Leer el archivo LAS desde la ruta temporal
            las_file = lasio.read(tmp_file_path, encoding='latin-1')
            df = las_file.df()
            
            # Mapear las columnas si la compañía es conocida
            df = map_columns_to_standard(df, company)
            all_dfs[file_name] = df
            
            st.success(f"'{file_name}' cargado y procesado exitosamente.")
        except Exception as e:
            st.error(f"Error al cargar o procesar '{file_name}': {e}")
            st.info(f"Asegúrate de que '{file_name}' sea un formato .LAS válido y no esté corrupto. El error detallado es: `{e}`")
        finally:
            # Asegurarse de eliminar el archivo temporal
            if tmp_file_path and os.path.exists(tmp_file_path):
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

def process_and_plot_pbu(uploaded_file_pbu):
    """Procesa un archivo LAS para el análisis de PBU y genera las gráficas."""
    las_data = None
    company = None
    tmp_file_path = None

    try:
        st.write(f"Procesando archivo: **{uploaded_file_pbu.name}**")
        file_bytes = uploaded_file_pbu.getvalue()
        
        # Guardar el archivo subido en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
        
        # Detectar compañía desde el archivo binario
        company = detect_compania(tmp_file_path)
        st.info(f"Compañía detectada: **{company}**")
        
        # Leer el archivo LAS desde la ruta temporal
        las_data = lasio.read(tmp_file_path, encoding='latin-1')
        st.success(f"'{uploaded_file_pbu.name}' cargado y procesado exitosamente.")
    
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo: {e}")
    finally:
        # Asegurarse de eliminar el archivo temporal
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    
    if las_data and company:
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
            
            # --- TERCER GRÁFICO: SPINNERS vs. ETIM (RESTABLECIDO) ---
            # Ahora el código busca las columnas estandarizadas, que ya han sido mapeadas
            # para ambas compañías (EXPRO y SLB).
            
            # Columnas necesarias después de la estandarización
            required_cols = ['ETIM', 'DEPTH', 'SPIN:1', 'SPIN:2', 'WFDE']
            if all(col in df.columns for col in required_cols):
                st.write("---")
                st.write("### Análisis de Velocidad de Spinner y Densidad de Fluido vs. Tiempo")
                
                # Extraer las columnas necesarias
                spinner_df = df[required_cols].copy()
                
                # Mostrar la tabla
                st.write("Tabla de datos de SPIN:1, SPIN:2, DEPTH y WFDE:")
                st.dataframe(spinner_df.head())
                st.dataframe(spinner_df.tail())
                
                # Crear el gráfico
                fig3, ax1 = plt.subplots(figsize=(12, 6))
                
                # Trazar SPIN:1 y SPIN:2 en el eje y principal
                ax1.plot(spinner_df['ETIM'], spinner_df['SPIN:1'], label='SPIN:1', color='blue')
                ax1.plot(spinner_df['ETIM'], spinner_df['SPIN:2'], label='SPIN:2', color='orange')
                ax1.set_xlabel('ETIM')
                ax1.set_ylabel('Velocidad del Spinner', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True)
                
                # Crear un eje y secundario para WFDE
                ax2 = ax1.twinx()
                ax2.plot(spinner_df['ETIM'], spinner_df['WFDE'], label='WFDE', color='green')
                ax2.set_ylabel('WFDE', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                
                # Agregar título y leyenda
                plt.title('SPIN:1, SPIN:2 y WFDE vs. ETIM')
                fig3.legend(loc="upper left")
                
                # Mostrar el gráfico en Streamlit
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.warning("El archivo no contiene las columnas necesarias para el análisis del spinner, incluso después de intentar el mapeo.")

        else:
            st.error("El archivo no contiene las columnas 'ETIM' o 'WPRE' necesarias para el análisis de PBU.")

# --- Contenedores de carga de archivos en la barra lateral ---
with st.sidebar:
    st.header("Carga de Archivos LAS")
    
    # Cargador para Análisis de Pasadas
    st.subheader("Análisis de Pasadas PLT")
    uploaded_files_plt = st.file_uploader(
        "Sube tus archivos de registro (.LAS)", 
        type=["las"], 
        accept_multiple_files=True,
        key="plt_uploader"
    )
    
    st.markdown("---")
    
    # Cargador para Análisis de PBU
    st.subheader("Análisis de PBU")
    uploaded_file_pbu = st.file_uploader(
        "Sube un archivo de registro (.LAS)",
        type=["las"],
        key="pbu_uploader"
    )

# --- Definición de pestañas ---
tab1, tab2 = st.tabs(["Análisis de Pasadas PLT", "Análisis de PBU"])

with tab1:
    st.header("Análisis de Pasadas y Control de Calidad")
    
    if uploaded_files_plt:
        # Procesamiento y clasificación
        PASADAS_SUBIENDO, PASADAS_BAJANDO = process_and_classify_files_plt(uploaded_files_plt)
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
        st.subheader("Visualización Interaciva de Registros PLT")

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
                    'Selecciona Columnas para Graficar (incluye los registros de Spinner):',
                    options=all_plot_columns,
                    default=all_plot_columns[:min(4, len(all_plot_columns))]
                )

                # --- LÓGICA PARA SELECCIONAR VELOCIDADES ---
                speeds_subiendo = sorted(df_subiendo['Speed'].dropna().unique().tolist()) if not df_subiendo.empty and 'Speed' in df_subiendo.columns else []
                speeds_bajando = sorted(df_bajando['Speed'].dropna().unique().tolist()) if not df_bajando.empty and 'Speed' in df_bajando.columns else []

                st.info(f"Velocidades detectadas para 'Subiendo': {speeds_subiendo}")
                st.info(f"Velocidades detectadas para 'Bajando': {speeds_bajando}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    speed_options_subiendo = ['Todas las Velocidades'] + [str(s) for s in speeds_subiendo]
                    selected_speed_subiendo_str = st.selectbox(
                        'Selecciona Velocidad (Subiendo):',
                        options=speed_options_subiendo,
                        key='selectbox_subiendo'
                    )
                    selected_speed_subiendo_value = int(selected_speed_subiendo_str) if selected_speed_subiendo_str != 'Todas las Velocidades' else None

                with col2:
                    speed_options_bajando = ['Todas las Velocidades'] + [str(s) for s in speeds_bajando]
                    selected_speed_bajando_str = st.selectbox(
                        'Selecciona Velocidad (Bajando):',
                        options=speed_options_bajando,
                        key='selectbox_bajando'
                    )
                    selected_speed_bajando_value = int(selected_speed_bajando_str) if selected_speed_bajando_str != 'Todas las Velocidades' else None

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
                        selected_speed=selected_speed_bajando_value,
                        min_dept=min_dept, 
                        max_dept=max_dept
                    )
    else:
        st.info("Por favor, sube uno o más archivos .LAS en la barra lateral para comenzar el análisis de pasadas.")

with tab2:
    st.header("Análisis de PBU (Pressure Build-Up)")

    if uploaded_file_pbu:
        process_and_plot_pbu(uploaded_file_pbu)
    else:
        st.info("Por favor, sube un archivo .LAS en la barra lateral para comenzar el análisis de PBU.")