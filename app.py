import pandas as pd
import numpy as np
import faker
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generar datos simulados
fake = faker.Faker()

def generate_data(num_entries):
    boat_ids = list(range(1, 101))  # Generar 100 identificadores únicos para barcos
    data = []
    
    for _ in range(num_entries):
        data.append({
            "date": fake.date_this_year(),
            "boat_id": np.random.choice(boat_ids),  # Elegir un identificador único al azar
            "fish_caught_kg": round(fake.random_int(min=100, max=1000) * fake.random.uniform(0.8, 1.2), 2),
            "water_depth_m": round(fake.random.uniform(1.5, 3.5), 2),
            "maintenance_cost_usd": round(fake.random_int(min=500, max=5000) * fake.random.uniform(0.9, 1.1), 2),
            "weather_conditions": fake.random_element(elements=("Sunny", "Rainy", "Stormy", "Cloudy")),
            "economic_indicator": round(fake.random_int(min=1000, max=5000) * fake.random.uniform(0.8, 1.2), 2),
            "environmental_impact": fake.random_element(elements=("Low", "Medium", "High")),
            "social_impact": fake.random_element(elements=("Positive", "Neutral", "Negative")),
            "legal_aspects": fake.random_element(elements=("Compliant", "Non-compliant")),
            "commercial_aspects": fake.random_element(elements=("Profitable", "Non-profitable")),
            "operational_efficiency": round(fake.random.uniform(0.5, 1.0), 2),
            "market_demand": round(fake.random_int(min=1000, max=10000) * fake.random.uniform(0.8, 1.2), 2),
            "optimal_capacity": round(fake.random_int(min=5000, max=20000) * fake.random.uniform(0.8, 1.2), 2),
            "infrastructure_quality": fake.random_element(elements=("High", "Medium", "Low")),
            "accessibility": fake.random_element(elements=("Good", "Average", "Poor")),
            "equipment_quality": fake.random_element(elements=("High", "Medium", "Low")),
            "traffic_management_system": fake.random_element(elements=("Advanced", "Basic", "None")),
            "customs_efficiency": fake.random_element(elements=("Efficient", "Inefficient")),
            "inspection_system": fake.random_element(elements=("Efficient", "Inefficient"))
        })
    
    return pd.DataFrame(data)

# Crear DataFrame con datos generados
data = generate_data(1000)

# Aplicación Streamlit
st.set_page_config(page_title="Análisis de Infraestructura Pesquera", layout="wide")
st.title("Análisis de Infraestructura Pesquera en Ushuaia y Almanza")

# Sidebar - Selección de opciones
st.sidebar.header("Opciones de Análisis")
analysis_option = st.sidebar.selectbox(
    "Seleccionar análisis:",
    ("Análisis Predictivo", "Indicadores de Sustentabilidad", "Diseño Funcional y Tecnológico")
)

# Visualización de datos
st.subheader("Vista previa de los datos generados:")
st.write(data.head())

# Función para filtrar datos por condiciones climáticas
def filter_by_weather_condition(weather_condition):
    return data[data['weather_conditions'] == weather_condition]

# Widget para filtrar por condiciones climáticas
if analysis_option == "Análisis Predictivo":
    st.subheader("Análisis Predictivo")
    X = data[['water_depth_m', 'maintenance_cost_usd', 'economic_indicator', 'operational_efficiency', 'market_demand']]
    y = data['fish_caught_kg']

    # Entrenar modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Predicciones
    data['predicted_fish_caught_kg'] = model.predict(X)

    # Visualización de resultados del análisis predictivo
    st.write("Resultados del análisis predictivo:")
    st.write(data[['date', 'boat_id', 'fish_caught_kg', 'predicted_fish_caught_kg']].head())

    # Gráfico de resultados
    fig, ax = plt.subplots()
    ax.scatter(data['water_depth_m'], data['fish_caught_kg'], color='blue', label='Actual')
    ax.scatter(data['water_depth_m'], data['predicted_fish_caught_kg'], color='red', label='Predicted')
    ax.set_xlabel('Profundidad del Agua (m)')
    ax.set_ylabel('Pescado Capturado (kg)')
    ax.legend()
    ax.set_title('Comparación de Capturas Reales vs Predichas')
    st.pyplot(fig)

elif analysis_option == "Indicadores de Sustentabilidad":
    st.subheader("Indicadores de Sustentabilidad")
    st.write(data[['date', 'economic_indicator', 'environmental_impact', 'social_impact']].head())

    # Gráfico de indicadores económicos
    fig, ax = plt.subplots()
    ax.hist(data['economic_indicator'], bins=30, color='green')
    ax.set_xlabel('Indicador Económico')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Indicadores Económicos')
    st.pyplot(fig)

    # Gráfico de impacto ambiental
    fig, ax = plt.subplots()
    data['environmental_impact'].value_counts().plot(kind='bar', color='brown', ax=ax)
    ax.set_xlabel('Impacto Ambiental')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución del Impacto Ambiental')
    st.pyplot(fig)

    # Gráfico de impacto social
    fig, ax = plt.subplots()
    data['social_impact'].value_counts().plot(kind='bar', color='blue', ax=ax)
    ax.set_xlabel('Impacto Social')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución del Impacto Social')
    st.pyplot(fig)

elif analysis_option == "Diseño Funcional y Tecnológico":
    st.subheader("Indicadores de Diseño Funcional, Físico y Tecnológico")
    st.write(data[['legal_aspects', 'commercial_aspects', 'operational_efficiency', 'market_demand', 'optimal_capacity',
                   'infrastructure_quality', 'accessibility', 'equipment_quality', 'traffic_management_system',
                   'customs_efficiency', 'inspection_system']].head())

    # Gráfico de indicadores de eficiencia operativa
    fig, ax = plt.subplots()
    ax.hist(data['operational_efficiency'], bins=30, color='purple')
    ax.set_xlabel('Eficiencia Operacional')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de la Eficiencia Operacional')
    st.pyplot(fig)

    # Gráfico de calidad de la infraestructura
    fig, ax = plt.subplots()
    data['infrastructure_quality'].value_counts().plot(kind='bar', color='orange', ax=ax)
    ax.set_xlabel('Calidad de Infraestructura')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de la Calidad de Infraestructura')
    st.pyplot(fig)

    # Gráfico de accesibilidad
    fig, ax = plt.subplots()
    data['accessibility'].value_counts().plot(kind='bar', color='red', ax=ax)
    ax.set_xlabel('Accesibilidad')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de la Accesibilidad')
    st.pyplot(fig)

# Widget para filtrar por condiciones climáticas (interactivo)
if analysis_option == "Análisis Predictivo":
    st.sidebar.subheader("Filtrar por Condiciones Climáticas")
    weather_condition = st.sidebar.selectbox('Seleccionar condiciones climáticas:', data['weather_conditions'].unique())
    filtered_data = filter_by_weather_condition(weather_condition)
    st.write(f"Datos filtrados por condiciones climáticas '{weather_condition}':")
    st.write(filtered_data.head())

# Información adicional y conclusión
st.sidebar.header("Información Adicional")
st.sidebar.markdown("""
La ingeniería pesquera desempeña un papel crucial en el desarrollo sostenible de las infraestructuras portuarias,
optimizando la eficiencia operativa y minimizando el impacto ambiental.
""")

st.sidebar.header("Conclusión")
st.sidebar.markdown("""
La integración de análisis de datos y ciencia de datos en la ingeniería pesquera es fundamental para
mejorar la planificación y ejecución de proyectos, asegurando un desarrollo portuario eficaz y sostenible.
""")

# Referencia al estudio y llamado a la acción
st.sidebar.header("Referencia")
st.sidebar.markdown("""
[Estudio completo en ResearchGate](https://www.researchgate.net/publication/382298455_PUERTOS_PESQUEROS_ASPECTOS_TECNICOS_PARA_EL_EMPLAZAMIENTO_Y_CONSTRUCCION_DE_UN_MUELLE_PARA_LA_PESCA_ARTESANAL_EN_USHUAIA)
""")

st.sidebar.header("Llamado a la Acción")
st.sidebar.markdown("""
Incorporar ingenieros pesqueros en la planificación de proyectos portuarios es esencial para optimizar
infraestructuras y promover un desarrollo más sostenible y eficiente.
""")
