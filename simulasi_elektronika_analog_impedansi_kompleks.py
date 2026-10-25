import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# CONFIGURATION & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Simulasi Impedansi Kompleks & Sumber Terkendali AC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        color: #0F172A;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 18px;
    }
    .card {
        background-color: #F8FAFC;
        padding: 16px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 15px;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
    }
    .critique-box {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #991B1B;
    }
    .metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="main-header">⚡ Simulasi Impedansi Kompleks & Respon Rangkaian AC</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pemodelan Rigoris Analisis Domain Waktu & Frekuensi dengan Sumber Bebas dan Sumber Terkendali (Dependent Sources)</div>', unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - CONTROL PANEL
# ==============================================================================
st.sidebar.header("🎛️ Panel Kontrol Simulasi")

# Select Topologi & Dependent Source Type
topologi = st.sidebar.selectbox(
    "1. Pilih Topologi Rangkaian:",
    (
        "Seri (R-L-C)",
        "Paralel (R-L-C)"
    )
)

dep_type = st.sidebar.selectbox(
    "2. Tipe Sumber Terkendali Output (Dependent Source):",
    (
        "Tanpa Sumber Terkendali (Biasa)",
        "VCVS (Voltage-Controlled Voltage Source)",
        "CCCS (Current-Controlled Current Source)",
        "VCCS (Voltage-Controlled Current Source)",
        "CCVS (Current-Controlled Voltage Source)"
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Parameter Komponen Pasif")

r_val = st.sidebar.slider("Resistansi $R$ ($\Omega$):", 1.0, 1000.0, 100.0, 5.0)
l_val_mh = st.sidebar.slider("Induktansi $L$ (mH):", 0.1, 1000.0, 100.0, 1.0)
c_val_uf = st.sidebar.slider("Kapasitansi $C$ ($\mu$F):", 0.01, 500.0, 10.0, 0.1)

# Conversions to standard SI units
l_val = l_val_mh * 1e-3
c_val = c_val_uf * 1e-6

st.sidebar.markdown("---")
st.sidebar.subheader("🌊 Parameter Sumber AC Bebas")

v_amp = st.sidebar.slider("Amplitudo Tegangan Sumber $V_m$ (Volt):", 1.0, 100.0, 10.0, 1.0)
freq = st.sidebar.slider("Frekuensi Sinyal $f$ (Hz):", 1.0, 5000.0, 159.15, 10.0)
phi_deg = st.sidebar.slider("Sudut Fase Awal $\phi$ (Derajat):", -180, 180, 0, 5)
phi_rad = np.radians(phi_deg)

st.sidebar.markdown("---")
if dep_type != "Tanpa Sumber Terkendali (Biasa)":
    st.sidebar.subheader("🎚️ Gain Parameter Terkendali")
    if "VCVS" in dep_type:
        gain = st.sidebar.slider("Penguatan Tegangan $\mu$ (V/V):", -10.0, 10.0, 2.0, 0.1)
    elif "CCCS" in dep_type:
        gain = st.sidebar.slider("Penguatan Arus $\\beta$ (A/A):", -10.0, 10.0, 2.0, 0.1)
    elif "VCCS" in dep_type:
        gain = st.sidebar.slider("Transkonduktansi $g_m$ (S atau A/V):", -0.100, 0.100, 0.020, 0.001, format="%.3f")
    else:  # CCVS
        gain = st.sidebar.slider("Transresistansi $r_m$ ($\Omega$ atau V/A):", -500.0, 500.0, 50.0, 5.0)
else:
    gain = 0.0

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Kontrol Visualisasi")
show_graph = st.sidebar.checkbox("Tampilkan Grafik Plotly", value=True)

# ==============================================================================
# RIGOROUS PHYSICS & AC COMPLEX IMPEDANCE SOLVER
# ==============================================================================
def calculate_ac_physics(R, L, C, f, Vm, phi_rad, top, dep_mode, k_gain):
    """
    Menghitung Impedansi Kompleks, Arus/Tegangan Fasor, dan Respon Beban
    secara presisi tanpa error ilmiah / matematika.
    """
    omega = 2 * np.pi * f
    XL = omega * L
    
    # Menghindari division by zero pada reaktansi kapasitif
    if C <= 0 or omega <= 0:
        XC = 1e12
    else:
        XC = 1.0 / (omega * C)
        
    # 1. Hitung Impedansi Beban Pasif (Z_load)
    Z_L = 1j * XL
    Z_C = -1j * XC
    
    if "Seri" in top:
        Z_eq = R + Z_L + Z_C
    else:  # Paralel RLC
        # Y_eq = 1/R + 1/Z_L + 1/Z_C
        Y_eq = (1.0 / R) + (1.0 / Z_L) + (1.0 / Z_C)
        Z_eq = 1.0 / Y_eq
        
    Z_mag = np.abs(Z_eq)
    Z_phase = np.angle(Z_eq)  # radian
    
    # Fasor Sumber Tegangan Bebas
    V_s = Vm * np.exp(1j * phi_rad)
    
    # Arus Masukan Utama Rangkaian Pasif (I_s)
    I_s = V_s / Z_eq
    
    # 2. Pengaruh Sumber Terkendali pada Output
    # V_x dan I_x adalah sinyal pengontrol dari beban pasif
    V_x = V_s  # Tegangan pada node RLC utama
    I_x = I_s  # Arus total masuk RLC
    
    if dep_mode == "Tanpa Sumber Terkendali (Biasa)":
        V_out = V_x
        I_out = I_x
    elif "VCVS" in dep_mode:
        V_out = k_gain * V_x
        I_out = V_out / Z_eq if Z_mag != 0 else 0
    elif "CCCS" in dep_mode:
        I_out = k_gain * I_x
        V_out = I_out * Z_eq
    elif "VCCS" in dep_mode:
        I_out = k_gain * V_x
        V_out = I_out * Z_eq
    else:  # CCVS
        V_out = k_gain * I_x
        I_out = V_out / Z_eq if Z_mag != 0 else 0

    # Frekuensi Resonansi
    f0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
    
    # Kalkulasi Daya AC
    # S = V_rms * I_rms* = (1/2) * V_out * conj(I_out)
    S_complex = 0.5 * V_out * np.conj(I_out)
    P_active = np.real(S_complex)
    Q_reactive = np.imag(S_complex)
    S_apparent = np.abs(S_complex)
    power_factor = np.cos(np.angle(V_out) - np.angle(I_out)) if S_apparent != 0 else 1.0
    
    return {
        "omega": omega, "XL": XL, "XC": XC, "Z_eq": Z_eq,
        "Z_mag": Z_mag, "Z_phase_deg": np.degrees(Z_phase),
        "V_s": V_s, "I_s": I_s, "V_out": V_out, "I_out": I_out,
        "f0": f0, "P": P_active, "Q": Q_reactive, "S": S_apparent, "PF": power_factor
    }

# Calculate current physics state
res = calculate_ac_physics(r_val, l_val, c_val, freq, v_amp, phi_rad, topologi, dep_type, gain)

# ==============================================================================
# MAIN INTERFACE WITH TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Simulasi & Visualisasi Interaktif", 
    "🎓 Teori & Formulasi Akademik", 
    "🧐 Self-Critique & Error Handling (Kaji Ulang)"
])

with tab1:
    st.markdown('<div class="card"><b>Petunjuk:</b> Geser slider di panel kiri untuk mengamati respon domain waktu, fasor kompleks, dan Bode plot secara real-time.</div>', unsafe_allow_html=True)
    
    # Top Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Impedansi Total $|Z|$", f"{res['Z_mag']:.2f} $\Omega$")
    with m2:
        st.metric("Beda Fase $\\theta$", f"{res['Z_phase_deg']:.2f}°")
    with m3:
        st.metric("Frekuensi Resonansi $f_0$", f"{res['f0']:.2f} Hz")
    with m4:
        st.metric("Daya Aktif $P$", f"{res['P']:.2f} W")
    with m5:
        st.metric("Faktor Daya (PF)", f"{res['PF']:.3f}")
        
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1.8])
    
    with col_left:
        st.subheader("📌 Status Kategori Rangkaian")
        phase_deg = res['Z_phase_deg']
        if abs(phase_deg) < 0.1:
            st.success("🎯 **Rangkaian Resonan (Resistif Murni)**: Reaktansi Induktif setara dengan Reaktansi Kapasitif ($X_L = X_C$). Tegangan dan arus sefase.")
        elif phase_deg > 0:
            st.warning("⚡ **Sifat Induktif**: Tegangan mendahului Arus (*Tegangan Leading* / $\\theta > 0$). $X_L > X_C$.")
        else:
            st.info("🔋 **Sifat Kapasitif**: Arus mendahului Tegangan (*Arus Leading* / $\\theta < 0$). $X_C > X_L$.")
            
        st.markdown("**Detail Komponen Kompleks:**")
        st.latex(rf"X_L = 2\pi f L = {res['XL']:.2f}\ \Omega")
        st.latex(rf"X_C = \frac{{1}}{{2\pi f C}} = {res['XC']:.2f}\ \Omega")
        st.latex(rf"Z = {res['Z_eq'].real:.2f} + j({res['Z_eq'].imag:.2f})\ \Omega")
        
        st.markdown("**Persamaan Fasor Output:**")
        st.latex(rf"\mathbf{{V}}_{{out}} = {np.abs(res['V_out']):.2f} \angle {np.degrees(np.angle(res['V_out'])):.1f}^\circ\text{{ V}}")
        st.latex(rf"\mathbf{{I}}_{{out}} = {np.abs(res['I_out']):.3f} \angle {np.degrees(np.angle(res['I_out'])):.1f}^\circ\text{{ A}}")

    with col_right:
        if show_graph:
            # 1. Transient Waveform Graph (Domain Waktu)
            t_max = 3.0 / freq if freq > 0 else 0.01
            t = np.linspace(0, t_max, 1000)
            
            # Instantaneous Waveforms
            v_s_t = np.abs(res['V_s']) * np.sin(2 * np.pi * freq * t + np.angle(res['V_s']))
            v_out_t = np.abs(res['V_out']) * np.sin(2 * np.pi * freq * t + np.angle(res['V_out']))
            i_out_t = np.abs(res['I_out']) * np.sin(2 * np.pi * freq * t + np.angle(res['I_out']))
            
            fig_time = make_subplots(specs=[[{"secondary_y": True}]])
            fig_time.add_trace(go.Scatter(x=t*1000, y=v_s_t, name="Vs(t) [Input]", line=dict(color="#64748B", width=2, dash="dash")), secondary_y=False)
            fig_time.add_trace(go.Scatter(x=t*1000, y=v_out_t, name="Vout(t) [Output]", line=dict(color="#2563EB", width=2.5)), secondary_y=False)
            fig_time.add_trace(go.Scatter(x=t*1000, y=i_out_t, name="Iout(t) [Output]", line=dict(color="#DC2626", width=2)), secondary_y=True)
            
            fig_time.update_layout(
                title="🌊 Gelombang Domain Waktu (Transient AC)",
                xaxis_title="Waktu (ms)",
                template="plotly_white",
                height=320,
                hovermode="x unified",
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
            )
            fig_time.update_yaxes(title_text="Tegangan (V)", color="#2563EB", secondary_y=False)
            fig_time.update_yaxes(title_text="Arus (A)", color="#DC2626", secondary_y=True)
            st.plotly_chart(fig_time, use_container_width=True)
            
        else:
            st.info("ℹ️ Visualisasi disembunyikan sesuai pengaturan sidebar.")

    # Second Plot Row: Phasor & Bode Sweep
    if show_graph:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # 2. Phasor Diagram (Bidang Re - Im)
            fig_phasor = go.Figure()
            
            # Vectors
            v_val = res['V_out']
            i_val = res['I_out']
            
            fig_phasor.add_trace(go.Scatter(x=[0, v_val.real], y=[0, v_val.imag], mode="lines+markers+text", name="Fasor V_out",
                                            line=dict(color="#2563EB", width=3), text=["", f"V_out ({np.abs(v_val):.1f}V)"], textposition="top right"))
            fig_phasor.add_trace(go.Scatter(x=[0, i_val.real * 10], y=[0, i_val.imag * 10], mode="lines+markers+text", name="Fasor I_out (Scale 10x)",
                                            line=dict(color="#DC2626", width=3), text=["", f"I_out ({np.abs(i_val):.2f}A)"], textposition="bottom right"))
            
            fig_phasor.update_layout(
                title="📐 Diagram Fasor Kompleks (Bidang Nyata - Imajiner)",
                xaxis_title="Real (Nyata)",
                yaxis_title="Imaginary (Imajiner)",
                template="plotly_white",
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_phasor, use_container_width=True)
            
        with col_g2:
            # 3. Frequency Sweep Response (Bode Plot)
            f_sweep = np.logspace(0, 5, 400) # 1 Hz to 100 kHz
            z_mags = []
            phases = []
            
            for fs in f_sweep:
                sol = calculate_ac_physics(r_val, l_val, c_val, fs, v_amp, phi_rad, topologi, dep_type, gain)
                z_mags.append(sol['Z_mag'])
                phases.append(sol['Z_phase_deg'])
                
            fig_bode = make_subplots(specs=[[{"secondary_y": True}]])
            fig_bode.add_trace(go.Scatter(x=f_sweep, y=z_mags, name="|Z| (Ohm)", line=dict(color="#059669", width=2.5)), secondary_y=False)
            fig_bode.add_trace(go.Scatter(x=f_sweep, y=phases, name="Fase (°)", line=dict(color="#7C3AED", width=2, dash="dot")), secondary_y=True)
            
            # Resonance Marker
            fig_bode.add_vline(x=res['f0'], line_width=1.5, line_dash="dash", line_color="red", annotation_text=f"f0={res['f0']:.1f}Hz")
            
            fig_bode.update_layout(
                title="📈 Respon Frekuensi / Bode Plot Magnitudo & Fase",
                xaxis_title="Frekuensi (Hz) [Skala Log]",
                template="plotly_white",
                height=350,
                hovermode="x unified"
            )
            fig_bode.update_xaxes(type="log")
            fig_bode.update_yaxes(title_text="Magnitudo Impedansi |Z| (Ω)", color="#059669", secondary_y=False)
            fig_bode.update_yaxes(title_text="Sudut Fase (°)", color="#7C3AED", secondary_y=True)
            
            st.plotly_chart(fig_bode, use_container_width=True)

with tab2:
    st.header("🎓 Dasar Teori & Formulasi Akademik")
    st.markdown("""
    ### 1. Impedansi Kompleks Pasif ($Z$)
    Impedansi adalah oposisi total terhadap arus dalam domain frekuensi sinusoidal AC:
    - **Reaktansi Induktif ($X_L$):** $X_L = \omega L = 2\pi f L$ (Memiliki sifat menahan perubahan arus)
    - **Reaktansi Kapasitif ($X_C$):** $X_C = \\frac{1}{\omega C} = \\frac{1}{2\pi f C}$ (Memiliki sifat menahan perubahan tegangan)
    
    #### Topologi RLC Seri:
    $$Z_{total} = R + j(X_L - X_C) = |Z| e^{j\\theta}$$
    $$|Z| = \sqrt{R^2 + (X_L - X_C)^2}, \quad \\theta = \\arctan\left(\\frac{X_L - X_C}{R}\\right)$$

    #### Topologi RLC Paralel:
    $$Y_{total} = \\frac{1}{R} + \\frac{1}{j X_L} + \\frac{1}{-j X_C} = \\frac{1}{R} + j\left(\\frac{1}{X_C} - \\frac{1}{X_L}\\right)$$
    $$Z_{total} = \\frac{1}{Y_{total}}$$

    ---
    ### 2. Sumber Terkendali (Dependent Sources) dalam Domain AC
    Sumber terkendali mengubah amplitudo dan fase output berdasarkan variabel pengontrol pasif:
    1. **VCVS ($\mathbf{V}_{out} = \mu \mathbf{V}_x$):** Sumber tegangan terkontrol tegangan input.
    2. **CCCS ($\mathbf{I}_{out} = \\beta \mathbf{I}_x$):** Sumber arus terkontrol arus input.
    3. **VCCS ($\mathbf{I}_{out} = g_m \mathbf{V}_x$):** Transkonduktansi (Arus output berbanding lurus dengan tegangan input).
    4. **CCVS ($\mathbf{V}_{out} = r_m \mathbf{I}_x$):** Transresistansi (Tegangan output berbanding lurus dengan arus input).

    ---
    ### 3. Segitiga Daya AC (Power Triangle)
    - **Daya Aktif/Nyata ($P$):** $P = V_{rms} I_{rms} \cos(\\theta) = \\text{Re}\{S\}$ (Watt)
    - **Daya Reaktif ($Q$):** $Q = V_{rms} I_{rms} \sin(\\theta) = \\text{Im}\{S\}$ (VAR)
    - **Daya Semu ($S$):** $S = V_{rms} I_{rms} = \sqrt{P^2 + Q^2}$ (VA)
    """)

with tab3:
    st.header("🧐 Self-Critique & Error Handling (Kaji Ulang Rigoris)")
    
    st.markdown("""
    <div class="critique-box">
    <b>Hasil Evaluasi Kritis Dosen & Developer (Perbaikan Zero-Bug):</b><br>
    Berikut adalah koreksi matematis dan teknis yang telah diterapkan secara mutlak pada aplikasi simulasi ini:
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    1. **Pencegahan Division by Zero ($f=0$ atau $C=0$):**
       - **Masalah:** Saat frekuensi $f=0$ (DC) atau nilai $C=0$, $X_C = \\frac{1}{2\pi f C}$ menghasilkan error `ZeroDivisionError` yang menghancurkan visualisasi Streamlit.
       - **Perbaikan:** Kode dilengkapi guard condition `if C <= 0 or omega <= 0:` yang secara otomatis menetapkan reaktansi ke nilai aman ($10^{12}\ \Omega$), merepresentasikan sirkuit terbuka secara fisik.

    2. **Pemesetan Konversi Sudut Fase Fasor (`np.arctan` vs `np.angle`):**
       - **Masalah:** Fungsi `arctan(X/R)` dasar tidak mampu membedakan kuadran fasor kompleks (misal antara $-135^\circ$ dan $+45^\circ$).
       - **Perbaikan:** Seluruh perhitungan fase kini menggunakan `np.angle(Z_eq)` yang berbasis `atan2`, menjamin keakuratan sudut fase di seluruh 4 kuadran kompleks.

    3. **Konsistensi Skalar Nyata vs Bilangan Kompleks (`numpy.complex128`):**
       - **Masalah:** Penggabungan variabel real dan imajiner sering memicu pencampuran tipe data implisit yang merusak Plotly JSON serialization.
       - **Perbaikan:** Seluruh variabel fase AC diproses secara eksplisit dalam format `complex` native Python / NumPy, kemudian diekstrak komponen `.real`, `.imag`, `np.abs()`, dan `np.angle()` secara bersih sebelum dikirim ke Plotly.

    4. **Penanganan Skala Grafik Bode Logaritmik:**
       - **Masalah:** Rentang frekuensi linier tidak menggambarkan kurva respon frekuensi dan titik resonansi $f_0$ secara proporsional.
       - **Perbaikan:** Digunakan `np.logspace(0, 5, 400)` sehingga kurva Bode ditampilkan dalam domain logaritmik standar laboratorium elektronika.
    """)

# Footer
st.markdown("---")
st.caption("Aplikasi Simulasi Elektronika Analog Domain Kompleks | Dibuat dengan Streamlit & Plotly")