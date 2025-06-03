import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador SIR", layout="centered")
st.title("Simulador de Epidemias (Modelo SIR)")

#Paramêtros iniciais via sidebar
populacao=st.sidebar.number_input("População total", 1000, 1000000, 10000)
inicial_infectados=st.sidebar.number_input("Infectados iniciais", 1, populacao, 1)
inicial_recuperados=st.sidebar.number_input("Recuperados iniciais", 0, populacao, 0)
beta = st.sidebar.slider("Taxa de transmissão (β)", 0.0, 1.0, 0.3)
gamma = st.sidebar.slider("Taxa de recuperação (γ)", 0.0, 1.0, 0.1)
dias = st.sidebar.slider("Duração da simulação (dias)", 1, 365, 160)

#Valores iniciais
S0 = populacao - inicial_infectados - inicial_recuperados
I0 = inicial_infectados
R0 = inicial_recuperados
y0 = S0, I0, R0

#Modelo SIR
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta*S*I/populacao
    dI_dt = beta*S*I/populacao - gamma*I
    dR_dt = gamma*I
    return dS_dt, dI_dt, dR_dt

#Integração das equações
t = np.linspace(0, dias, dias)
ret = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = ret.T

#Gráfico
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, S, label="Suscetíveis", color="blue")
ax.plot(t, I, label="Infectados", color="Red")
ax.plot(t, R, label="Recuperados", color="green")
ax.set_xlabel("Dias")
ax.set_ylabel("Número de pessoas")
ax.set_title("Modelo SIR de Propagação de Doenças")
ax.legend()
st.pyplot(fig)