"""MediScan AI — Ultimate Premium Edition | B R Sathvik · AMC Engineering College"""
import os, numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
import cv2, streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="MediScan AI — Chest X-Ray Analysis", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root{--bg-base:#030b18;--bg-elevated:#070f1f;--bg-card:#0a1628;--border:rgba(255,255,255,0.06);--border-bright:rgba(255,255,255,0.10);--cyan:#06b6d4;--cyan-dim:rgba(6,182,212,0.12);--indigo:#818cf8;--indigo-dim:rgba(129,140,248,0.12);--red:#ef4444;--green:#10b981;--amber:#f59e0b;--text-primary:#e2e8f0;--text-secondary:#64748b;--text-muted:#334155;--text-dim:#8899aa;--radius-sm:8px;--radius-md:12px;--radius-lg:18px;--radius-xl:24px}
*,*::before,*::after{box-sizing:border-box}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],.stDeployButton,[data-testid="collapsedControl"]{display:none!important}
.stApp{background:var(--bg-base);font-family:'Inter',-apple-system,sans-serif;color:var(--text-primary);min-height:100vh}
.stApp::before{content:'';position:fixed;top:0;left:0;right:0;bottom:0;background:radial-gradient(ellipse 90% 55% at 5% -5%,rgba(6,182,212,0.11) 0%,transparent 55%),radial-gradient(ellipse 65% 50% at 95% 95%,rgba(99,102,241,0.09) 0%,transparent 55%),radial-gradient(ellipse 45% 35% at 50% 50%,rgba(6,182,212,0.03) 0%,transparent 65%),linear-gradient(180deg,#030b18 0%,#020810 100%);pointer-events:none;z-index:0}
.block-container{max-width:1320px!important;padding:0 2.5rem 6rem!important;position:relative;z-index:1}
.hero-wrap{text-align:center;padding:3.5rem 2rem 2.5rem;position:relative}
.hero-badge{display:inline-flex;align-items:center;gap:.5rem;background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.18);border-radius:100px;padding:.32rem 1.1rem;font-size:.62rem;font-family:'Space Mono',monospace;color:var(--cyan);letter-spacing:.16em;text-transform:uppercase;margin-bottom:1.4rem;backdrop-filter:blur(8px)}
.hero-badge-dot{width:5px;height:5px;background:var(--cyan);border-radius:50%;flex-shrink:0;animation:badge-pulse 2.2s ease-in-out infinite}
@keyframes badge-pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.75)}}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(3rem,5.5vw,5.2rem);font-weight:800;letter-spacing:-.035em;line-height:1.02;margin-bottom:1rem;background:linear-gradient(140deg,#f1f5f9 0%,#67e8f9 35%,#a5b4fc 70%,#f1f5f9 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;background-size:200% 200%;animation:gradient-shift 8s ease-in-out infinite}
@keyframes gradient-shift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.hero-sub{font-size:.95rem;color:var(--text-secondary);max-width:500px;margin:0 auto .5rem;line-height:1.7;font-weight:400}
.hero-divider{width:60px;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);margin:1.5rem auto 0}
.disclaimer-bar{display:flex;align-items:flex-start;gap:.8rem;background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.14);border-left:2px solid rgba(245,158,11,0.5);border-radius:var(--radius-md);padding:.9rem 1.3rem;font-size:.76rem;color:#7c6a3a;margin-bottom:2rem;line-height:1.6;backdrop-filter:blur(4px)}
.stTabs [data-baseweb="tab-list"]{background:rgba(7,15,31,0.95)!important;border:1px solid rgba(255,255,255,0.06)!important;border-radius:var(--radius-md)!important;padding:5px!important;gap:3px!important;margin-bottom:2rem!important;backdrop-filter:blur(16px)!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#8899aa!important;border-radius:var(--radius-sm)!important;font-family:'Inter',sans-serif!important;font-size:.82rem!important;font-weight:500!important;padding:.65rem 1.6rem!important;transition:all .25s ease!important;border:none!important;letter-spacing:.01em!important}
.stTabs [data-baseweb="tab"]:hover{color:var(--text-secondary)!important;background:rgba(255,255,255,0.03)!important}
.stTabs [aria-selected="true"]{background:rgba(6,182,212,0.09)!important;color:var(--cyan)!important;border:1px solid rgba(6,182,212,0.20)!important;box-shadow:0 2px 12px rgba(6,182,212,0.08)!important}
.stTabs [data-baseweb="tab-border"],.stTabs [data-baseweb="tab-highlight"]{display:none!important}
.stTabs [data-baseweb="tab-panel"]{padding:0!important}
.sec-lbl{font-family:'Space Mono',monospace;font-size:.6rem;font-weight:700;color:var(--text-dim);text-transform:uppercase;letter-spacing:.2em;margin-bottom:.85rem;display:flex;align-items:center;gap:.7rem}
.sec-lbl::after{content:'';flex:1;height:1px;background:rgba(255,255,255,0.04)}
.glass-panel{background:rgba(10,22,40,0.8);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.8rem;backdrop-filter:blur(20px);position:relative;overflow:hidden}
.glass-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent 0%,rgba(6,182,212,0.2) 50%,transparent 100%)}
[data-testid="stFileUploader"]{border:1.5px dashed rgba(6,182,212,0.15)!important;border-radius:var(--radius-md)!important;background:rgba(6,182,212,0.02)!important;transition:all .3s ease!important;padding:.5rem!important}
[data-testid="stFileUploader"]:hover{border-color:rgba(6,182,212,0.38)!important;background:rgba(6,182,212,0.05)!important}
[data-testid="stFileUploaderDropzone"]{background:transparent!important}
[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] span{color:var(--text-secondary)!important;font-size:.8rem!important}
.result-card{border-radius:var(--radius-lg);padding:2rem 1.75rem;text-align:center;position:relative;overflow:hidden;margin-bottom:1.5rem}
.result-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;border-radius:var(--radius-lg) var(--radius-lg) 0 0}
.result-pneumonia{background:linear-gradient(145deg,rgba(239,68,68,0.13) 0%,rgba(185,28,28,0.06) 60%,rgba(127,29,29,0.03) 100%);border:1px solid rgba(239,68,68,0.22);animation:glow-red 3.5s ease-in-out infinite}
.result-pneumonia::before{background:linear-gradient(90deg,transparent,rgba(239,68,68,0.3),transparent)}
.result-normal{background:linear-gradient(145deg,rgba(16,185,129,0.13) 0%,rgba(5,150,105,0.06) 60%,rgba(6,78,59,0.03) 100%);border:1px solid rgba(16,185,129,0.22);animation:glow-green 3.5s ease-in-out infinite}
.result-normal::before{background:linear-gradient(90deg,transparent,rgba(16,185,129,0.3),transparent)}
@keyframes glow-red{0%,100%{box-shadow:0 0 30px rgba(239,68,68,0.06),inset 0 0 30px rgba(239,68,68,0.02)}50%{box-shadow:0 0 60px rgba(239,68,68,0.16),inset 0 0 40px rgba(239,68,68,0.04)}}
@keyframes glow-green{0%,100%{box-shadow:0 0 30px rgba(16,185,129,0.06),inset 0 0 30px rgba(16,185,129,0.02)}50%{box-shadow:0 0 60px rgba(16,185,129,0.16),inset 0 0 40px rgba(16,185,129,0.04)}}
.result-icon{font-size:3rem;display:block;margin-bottom:.65rem}
.result-label{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;letter-spacing:.08em;margin-bottom:.4rem}
.result-conf{font-family:'Space Mono',monospace;font-size:.92rem}
.prob-container{background:rgba(7,15,31,0.9);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.4rem 1.6rem;margin-bottom:1.5rem;position:relative;overflow:hidden}
.prob-container::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--border-bright),transparent)}
.prob-title{font-family:'Space Mono',monospace;font-size:.6rem;text-transform:uppercase;letter-spacing:.18em;color:var(--text-dim);margin-bottom:1.1rem}
.prob-item{margin-bottom:1.1rem}
.prob-item:last-child{margin-bottom:0}
.prob-header-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:.45rem}
.prob-cls-name{font-family:'JetBrains Mono',monospace;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em}
.prob-cls-normal{color:#34d399}
.prob-cls-pneumonia{color:#f87171}
.prob-pct{font-family:'Space Mono',monospace;font-size:.75rem;font-weight:700}
.prob-pct-normal{color:var(--green)}
.prob-pct-pneumonia{color:var(--red)}
.prob-track{width:100%;height:8px;background:rgba(255,255,255,0.04);border-radius:100px;overflow:hidden;position:relative}
.prob-fill{height:100%;border-radius:100px;position:relative;transition:width .8s cubic-bezier(0.4,0,0.2,1)}
.prob-fill::after{content:'';position:absolute;top:0;right:0;width:4px;height:100%;background:rgba(255,255,255,0.4);border-radius:100px;filter:blur(2px)}
.prob-fill-normal{background:linear-gradient(90deg,#047857 0%,#10b981 70%,#34d399 100%)}
.prob-fill-pneumonia{background:linear-gradient(90deg,#991b1b 0%,#ef4444 70%,#fca5a5 100%)}
.chips-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:1.1rem}
.info-chip{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:var(--radius-sm);padding:.65rem .9rem;transition:border-color .25s}
.info-chip:hover{border-color:rgba(6,182,212,0.15)}
.chip-label{font-family:'Space Mono',monospace;font-size:.55rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.2rem}
.chip-value{font-size:.8rem;color:var(--text-secondary);font-weight:500;font-family:'JetBrains Mono',monospace}
.heatmap-legend{display:flex;align-items:center;justify-content:center;gap:.65rem;margin-top:.75rem}
.legend-bar{width:100px;height:5px;border-radius:100px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#059669,#ca8a04,#dc2626)}
.legend-text{font-family:'Space Mono',monospace;font-size:.58rem;color:var(--text-dim);letter-spacing:.06em}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:480px;border:1.5px dashed rgba(255,255,255,0.05);border-radius:var(--radius-lg);color:var(--text-dim);font-family:'Space Mono',monospace;font-size:.75rem;text-align:center;line-height:2.4;gap:.25rem}
.empty-icon{font-size:2.5rem;opacity:.2;margin-bottom:.5rem}
.stButton>button{background:rgba(6,182,212,0.06)!important;border:1px solid rgba(6,182,212,0.18)!important;color:#67e8f9!important;border-radius:var(--radius-sm)!important;font-family:'Inter',sans-serif!important;font-weight:500!important;font-size:.78rem!important;padding:.6rem 1.1rem!important;transition:all .22s ease!important;width:100%!important;letter-spacing:.015em!important;position:relative!important;overflow:hidden!important}
.stButton>button:hover{background:rgba(6,182,212,0.13)!important;border-color:rgba(6,182,212,0.42)!important;transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(6,182,212,0.12)!important;color:#a5f3fc!important}
.stButton>button:active{transform:translateY(0)!important}
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2.5rem}
.metric-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-md);height:130px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;position:relative;overflow:hidden;transition:border-color .3s ease,box-shadow .3s ease;cursor:default}
.metric-card:hover{border-color:rgba(6,182,212,0.22);box-shadow:0 0 24px rgba(6,182,212,0.06)}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.2),transparent);opacity:0;transition:opacity .3s}
.metric-card:hover::before{opacity:1}
.metric-card::after{content:'';position:absolute;bottom:0;left:20%;right:20%;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.15),transparent)}
.mc-label{font-family:'Space Mono',monospace;font-size:.55rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.15em;margin-bottom:.5rem}
.mc-value{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;line-height:1;margin-bottom:.35rem;background:linear-gradient(135deg,#67e8f9 0%,var(--indigo) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.mc-sub{font-size:.62rem;color:var(--text-dim);font-family:'Inter',sans-serif}
.stack-table{width:100%;border-collapse:collapse;border-radius:var(--radius-md);overflow:hidden;border:1px solid var(--border);margin-bottom:2rem}
.stack-table th{background:rgba(6,182,212,0.07);color:var(--cyan);font-family:'Space Mono',monospace;font-size:.62rem;text-transform:uppercase;letter-spacing:.12em;padding:.9rem 1.2rem;text-align:left;border:none}
.stack-table td{color:var(--text-secondary);padding:.85rem 1.2rem;font-size:.82rem;border:none;border-top:1px solid var(--border);font-family:'Inter',sans-serif;transition:background .2s}
.stack-table tr:hover td{background:rgba(255,255,255,0.015)}
.stack-table td:first-child{color:var(--text-muted);font-weight:500}
.stack-table td .tag{display:inline-block;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.15);border-radius:100px;padding:.1rem .55rem;font-size:.68rem;color:var(--cyan);font-family:'JetBrains Mono',monospace;margin-left:.4rem}
.perf-badge{display:inline-flex;align-items:center;gap:.4rem;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.15);border-radius:100px;padding:.25rem .75rem;font-family:'Space Mono',monospace;font-size:.62rem;color:#34d399;letter-spacing:.06em;margin:0 .25rem}
.about-banner{background:linear-gradient(135deg,rgba(6,182,212,0.06) 0%,rgba(99,102,241,0.06) 50%,rgba(6,182,212,0.03) 100%);border:1px solid rgba(6,182,212,0.12);border-radius:var(--radius-xl);padding:2rem 2.5rem;margin-bottom:2rem;display:flex;align-items:center;justify-content:space-between;gap:2rem;position:relative;overflow:hidden}
.about-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.3),rgba(99,102,241,0.3),transparent)}
.about-title{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:var(--text-primary);margin-bottom:.4rem}
.about-desc{font-size:.82rem;color:var(--text-secondary);line-height:1.65;max-width:480px}
.about-stats{display:flex;gap:1.5rem;flex-shrink:0}
.about-stat{text-align:center}
.about-stat-val{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,var(--cyan),var(--indigo));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;display:block}
.about-stat-lbl{font-size:.6rem;color:var(--text-dim);font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:.1em}
.chat-header{background:rgba(7,15,31,0.9);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.25rem 1.6rem;margin-bottom:1.5rem;display:flex;align-items:center;justify-content:space-between}
.chat-header-left{display:flex;align-items:center;gap:.75rem}
.chat-status-dot{width:8px;height:8px;background:var(--green);border-radius:50%;box-shadow:0 0 0 3px rgba(16,185,129,0.15);animation:status-pulse 2s infinite;flex-shrink:0}
@keyframes status-pulse{0%,100%{box-shadow:0 0 0 3px rgba(16,185,129,0.15)}50%{box-shadow:0 0 0 6px rgba(16,185,129,0.05)}}
.chat-title{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:var(--text-primary)}
.chat-subtitle{font-family:'Space Mono',monospace;font-size:.6rem;color:var(--text-dim);letter-spacing:.06em;margin-top:.1rem}
.chat-model-tag{background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.14);border-radius:100px;padding:.25rem .75rem;font-family:'Space Mono',monospace;font-size:.58rem;color:var(--cyan);letter-spacing:.08em}
[data-testid="stChatMessage"]{background:rgba(7,15,31,0.75)!important;border:1px solid var(--border)!important;border-radius:var(--radius-md)!important;margin-bottom:.8rem!important;backdrop-filter:blur(8px)!important;transition:border-color .2s!important}
[data-testid="stChatMessage"]:hover{border-color:var(--border-bright)!important}
[data-testid="stChatMessage"] p{color:var(--text-secondary)!important;font-size:.85rem!important;line-height:1.65!important}
[data-testid="stChatInput"]{background:rgba(7,15,31,0.9)!important;border:1px solid rgba(255,255,255,0.08)!important;border-radius:var(--radius-md)!important;backdrop-filter:blur(8px)!important}
[data-testid="stChatInput"] textarea{color:var(--text-secondary)!important;font-family:'Inter',sans-serif!important;font-size:.85rem!important}
[data-testid="stChatInput"] textarea::placeholder{color:var(--text-dim)!important}
.stChatFloatingInputContainer{background:transparent!important;border-top:none!important;padding-top:0!important}
.built-by-card{background:rgba(7,15,31,0.85);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.6rem 2rem;display:flex;align-items:center;justify-content:space-between;gap:2rem;margin-top:2rem;position:relative;overflow:hidden;transition:border-color .3s}
.built-by-card:hover{border-color:rgba(6,182,212,0.15)}
.built-by-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.15),transparent)}
.by-avatar{width:44px;height:44px;background:linear-gradient(135deg,var(--cyan-dim),var(--indigo-dim));border:1px solid rgba(6,182,212,0.2);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0}
.by-info{flex:1}
.by-name{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:.2rem}
.by-role{font-size:.75rem;color:var(--text-muted)}
.by-links{display:flex;gap:.75rem;flex-shrink:0}
.by-link{display:inline-flex;align-items:center;gap:.35rem;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.14);border-radius:var(--radius-sm);padding:.4rem .85rem;font-family:'Space Mono',monospace;font-size:.65rem;color:var(--cyan);text-decoration:none;letter-spacing:.05em;transition:all .2s}
.by-link:hover{background:rgba(6,182,212,0.12);border-color:rgba(6,182,212,0.35);color:#a5f3fc}
[data-testid="stImage"] img{border-radius:var(--radius-md)!important;display:block!important}
[data-testid="stAlert"]{background:rgba(7,15,31,0.6)!important;border:1px solid var(--border)!important;border-radius:var(--radius-md)!important;color:var(--text-secondary)!important;font-size:.8rem!important;backdrop-filter:blur(8px)!important}
[data-testid="stCaptionContainer"] p{color:var(--text-dim)!important;font-size:.68rem!important;font-family:'Space Mono',monospace!important;text-align:center!important;margin-top:.35rem!important}
.stSpinner>div{border-top-color:var(--cyan)!important}
hr{border:none!important;border-top:1px solid rgba(255,255,255,0.05)!important;margin:1.75rem 0!important}
p{color:var(--text-secondary)!important;font-size:.85rem!important;line-height:1.65!important}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--text-primary)!important}
h3{font-size:1rem!important;font-weight:700!important}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(6,182,212,0.2);border-radius:100px}
::-webkit-scrollbar-thumb:hover{background:rgba(6,182,212,0.4)}
</style>""", unsafe_allow_html=True)

# ── Constants ──
MODEL_PATH = "mediscan_best.pth"
DEVICE     = torch.device("cpu")
CLASSES    = ["NORMAL", "PNEUMONIA"]
IMG_SIZE   = 224
GROQ_KEY = os.environ.get("GROQ_KEY", "YOUR_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
TRANSFORM  = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

CLINICAL_KB = """Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough, fever, chills, and difficulty breathing. Bacteria, viruses and fungi can cause pneumonia.
Chest X-ray findings in Pneumonia: Consolidation (opacification of lung parenchyma), Air bronchograms (visible bronchi within consolidated lung tissue), Pleural effusion may accompany severe pneumonia, Bilateral infiltrates suggest atypical or viral pneumonia, Lobar consolidation usually indicates bacterial pneumonia.
Normal Chest X-ray: Clear lung fields bilaterally, Normal cardiac silhouette (cardiothoracic ratio < 0.5), Sharp costophrenic angles, No mediastinal widening.
WHO Treatment Guidelines: Mild pneumonia: Amoxicillin 500mg TDS x 5 days. Severe pneumonia: IV benzylpenicillin or ampicillin + gentamicin. Atypical pneumonia: Azithromycin or doxycycline. Supportive care: Hydration, oxygen if SpO2 < 94%, antipyretics.
Risk Factors: Age extremes (< 2 years, > 65 years), Smoking and alcohol, COPD/asthma, Immunocompromised states (HIV, chemotherapy), Hospitalization and mechanical ventilation.
Pediatric Pneumonia: Most common cause: Streptococcus pneumoniae (bacterial), RSV (viral). Viral pneumonia more common under 5 years. Penicillin first-line for bacterial pneumonia in children. Hospitalization if SpO2 < 90%, severe distress.
Grad-CAM: Explainability technique highlighting regions important for model prediction. Red/warm = high importance; blue/cool = low importance. Should highlight lung fields, consolidations, infiltrates.
Emergency care for pneumonia: Difficulty breathing, Chest pain, Confusion, SpO2 below 92%, Coughing blood, High fever not responding to medication.
AI Limitations: AI should assist not replace clinical judgment. Models may not generalize across populations. Always correlate AI findings with clinical history."""

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    model = models.densenet121(weights=None)
    in_f  = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4),nn.Linear(in_f,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_rag():
    docs = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=60).create_documents([CLINICAL_KB])
    emb  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device":"cpu"})
    return FAISS.from_documents(docs, emb)

class GradCAM:
    def __init__(self,model,layer):
        self.model=model;self.G=None;self.A=None
        layer.register_forward_hook(lambda m,i,o:setattr(self,'A',o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go:setattr(self,'G',go[0].detach()))
    def generate(self,t,ci):
        self.model.zero_grad();o=self.model(t);o[0,ci].backward()
        w=self.G.mean(dim=[2,3],keepdim=True)
        c=torch.relu((w*self.A).sum(dim=1,keepdim=True)).squeeze().numpy()
        c=cv2.resize(c,(IMG_SIZE,IMG_SIZE));return (c-c.min())/(c.max()-c.min()+1e-8)

def build_overlay(pil,cam):
    orig=np.array(pil.resize((IMG_SIZE,IMG_SIZE)).convert("RGB"))
    h=cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    return Image.fromarray((0.55*orig+0.45*h).astype(np.uint8))

def run_inference(model,pil):
    t=TRANSFORM(pil.convert("RGB")).unsqueeze(0);t.requires_grad_(True)
    gc=GradCAM(model,model.features.denseblock4.denselayer16.conv2)
    with torch.set_grad_enabled(True):
        logits=model(t);probs=torch.softmax(logits,dim=1)[0].detach().numpy()
        pred=int(probs.argmax());cam=gc.generate(t,pred)
    return pred,probs,build_overlay(pil,cam)

def get_rag_answer(q,store):
    if not GROQ_KEY: return "⚠️ GROQ_API_KEY not configured."
    ctx="\n\n".join([d.page_content for d in store.similarity_search(q,k=3)])
    r=Groq(api_key=GROQ_KEY).chat.completions.create(model=GROQ_MODEL,messages=[
        {"role":"system","content":"You are a clinical AI assistant for educational purposes only. Answer using only the provided context. Be concise. End with: 'Consult a qualified physician for medical advice.'"},
        {"role":"user","content":f"Context:\n{ctx}\n\nQuestion: {q}"}],max_tokens=450,temperature=0.3)
    return r.choices[0].message.content



# ── HTML helpers ──
def sec_label(t): return f'<div class="sec-lbl">{t}</div>'

def render_result_card(label,conf):
    if label=="PNEUMONIA":
        return f'<div class="result-card result-pneumonia"><span class="result-icon">🫁</span><div class="result-label" style="color:#ef4444">{label}</div><div class="result-conf" style="color:#f87171;font-family:Space Mono,monospace;font-size:.9rem;margin-top:.3rem">{conf:.1f}% confidence</div></div>'
    return f'<div class="result-card result-normal"><span class="result-icon">✅</span><div class="result-label" style="color:#10b981">{label}</div><div class="result-conf" style="color:#34d399;font-family:Space Mono,monospace;font-size:.9rem;margin-top:.3rem">{conf:.1f}% confidence</div></div>'

def render_prob_bars(probs):
    n,p=float(probs[0])*100,float(probs[1])*100
    return f'''<div class="prob-container"><div class="prob-title">Class Probability Distribution</div>
<div class="prob-item"><div class="prob-header-row"><span class="prob-cls-name prob-cls-normal">● Normal</span><span class="prob-pct prob-pct-normal">{n:.2f}%</span></div><div class="prob-track"><div class="prob-fill prob-fill-normal" style="width:{n}%"></div></div></div>
<div class="prob-item"><div class="prob-header-row"><span class="prob-cls-name prob-cls-pneumonia">● Pneumonia</span><span class="prob-pct prob-pct-pneumonia">{p:.2f}%</span></div><div class="prob-track"><div class="prob-fill prob-fill-pneumonia" style="width:{p}%"></div></div></div></div>'''

def render_info_chips(pil):
    w,h=pil.size
    return f'<div class="chips-grid"><div class="info-chip"><div class="chip-label">Resolution</div><div class="chip-value">{w} × {h} px</div></div><div class="info-chip"><div class="chip-label">Color Mode</div><div class="chip-value">{pil.mode}</div></div><div class="info-chip"><div class="chip-label">Model</div><div class="chip-value">DenseNet-121</div></div><div class="info-chip"><div class="chip-label">Inference</div><div class="chip-value">CPU · PyTorch</div></div></div>'

def render_metric_cards():
    cards=[("Test AUC-ROC","0.96+","Primary metric"),("Test Accuracy","90.2%","Held-out test set"),("Architecture","DN-121","DenseNet 121-layer"),("Training Images","5,863","NIH / Kaggle")]
    inner="".join(f'<div class="metric-card"><div class="mc-label">{l}</div><div class="mc-value">{v}</div><div class="mc-sub">{s}</div></div>' for l,v,s in cards)
    return f'<div class="metrics-grid">{inner}</div>'

def render_stack_table():
    rows=[("Backbone","DenseNet-121","ImageNet pretrained"),("Fine-tuned Layers","DenseBlock4 + Classifier Head","~2.4M parameters"),("Explainability","Grad-CAM","Gradient-weighted heatmaps"),("RAG Embeddings","all-MiniLM-L6-v2","sentence-transformers"),("Vector Store","FAISS","In-memory index"),("LLM","LLaMA-3.3-70B Versatile","via Groq API"),("Framework","PyTorch + Streamlit","Python 3.10+")]
    tbody="".join(f'<tr><td>{c}</td><td>{t} <span class="tag">{n}</span></td></tr>' for c,t,n in rows)
    return f'<table class="stack-table"><thead><tr><th>Component</th><th>Technology</th></tr></thead><tbody>{tbody}</tbody></table>'

QUICK_QS=["What does consolidation look like on a chest X-ray?","How is bacterial pneumonia treated according to WHO?","What is Grad-CAM and why is it important?","What are the risk factors for pneumonia?","When should a pneumonia patient go to the ER?"]

# ══════════════ PAGE ══════════════
st.markdown("""<div class="hero-wrap"><div class="hero-badge"><span class="hero-badge-dot"></span> AI · Medical Imaging · Explainable AI</div><div class="hero-title">MediScan AI</div><div class="hero-sub">Chest X-Ray analysis powered by DenseNet-121 &amp; Grad-CAM — built for education, research and clinical understanding</div><div class="hero-divider"></div></div>""",unsafe_allow_html=True)
st.markdown("""<div class="disclaimer-bar"><span>⚠</span><span><strong>Medical Disclaimer:</strong> This tool is for <strong>educational and research purposes only</strong>. It is <strong>NOT</strong> a substitute for professional clinical diagnosis. Predictions may contain errors. Always consult a qualified physician before making any medical decisions.</span></div>""",unsafe_allow_html=True)

tab1,tab2,tab3=st.tabs(["🔬  X-Ray Analysis","💬  Clinical Assistant","📊  About & Metrics"])

# ── TAB 1 ──
with tab1:
    model=load_model()
    if model is None:
        st.error("❌ Model file `mediscan_best.pth` not found. Please add it to the project root.")
        st.stop()
    L,R=st.columns([1,1],gap="small")
    with L:
        st.markdown(sec_label("Upload Chest X-Ray"),unsafe_allow_html=True)
        uploaded=st.file_uploader("Drag & drop chest X-ray (JPEG or PNG)",type=["jpg","jpeg","png"],label_visibility="collapsed")
        st.markdown("""
        <div style="background:rgba(6,182,212,0.04);border:1px solid rgba(6,182,212,0.12);border-radius:10px;
                    padding:.75rem 1rem;font-family:'Space Mono',monospace;font-size:.65rem;
                    color:#1e3a5f;letter-spacing:.04em;line-height:1.7;margin-top:.5rem">
            💡 Upload a real chest X-ray (JPEG/PNG) from the Kaggle dataset or your own files.<br>
            Supported: frontal PA/AP view chest X-rays · Max 200MB
        </div>""", unsafe_allow_html=True)
        pil=None
        if uploaded:
            pil=Image.open(uploaded)
        if pil:
            st.markdown("<div style='height:.9rem'></div>",unsafe_allow_html=True)
            st.markdown(sec_label("Input X-Ray"),unsafe_allow_html=True)
            st.image(pil.resize((IMG_SIZE, IMG_SIZE)),use_container_width=True)
            st.markdown(render_info_chips(pil),unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state"><span class="empty-icon">🫁</span><span>Drop a chest X-ray image above<br>to begin AI analysis</span></div>',unsafe_allow_html=True)
    with R:
        if pil:
            with st.spinner("Running DenseNet-121 inference · Generating Grad-CAM…"):
                pred,probs,ov=run_inference(model,pil)
            lbl=CLASSES[pred];conf=float(probs[pred])*100
            st.markdown(render_result_card(lbl,conf),unsafe_allow_html=True)
            st.markdown(render_prob_bars(probs),unsafe_allow_html=True)
            st.markdown(sec_label("Grad-CAM Heatmap"),unsafe_allow_html=True)
            st.image(ov,use_container_width=True)
            st.markdown('<div class="heatmap-legend"><span class="legend-text">Low</span><div class="legend-bar"></div><span class="legend-text">High attention · Red = model focus regions</span></div>',unsafe_allow_html=True)
            st.caption("Heatmap should highlight lung fields. Attention on background areas may indicate lower reliability.")
        else:
            st.markdown('<div class="empty-state" style="min-height:620px"><span class="empty-icon">📊</span><span>Analysis results will appear here<br>once you upload a chest X-ray</span></div>',unsafe_allow_html=True)

# ── TAB 2 ──
with tab2:
    store=load_rag()
    st.markdown("""<div class="chat-header"><div class="chat-header-left"><div class="chat-status-dot"></div><div><div class="chat-title">Clinical Knowledge Assistant</div><div class="chat-subtitle">Powered by WHO clinical guidelines · For education only</div></div></div><div class="chat-model-tag">LLaMA-3.3-70B · Groq</div></div>""",unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages=[]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.write(m["content"])
    st.markdown(sec_label("Quick Questions"),unsafe_allow_html=True)
    for row,qs in [([0,1,2],st.columns(3)),([3,4],st.columns(2))]:
        for i,col in zip(row,qs):
            with col:
                if st.button(QUICK_QS[i],key=f"qq_{i}",use_container_width=True):
                    st.session_state.pending_q=QUICK_QS[i]
    st.markdown("<div style='height:.5rem'></div>",unsafe_allow_html=True)
    prompt=st.chat_input("Ask a clinical question about chest X-rays or pneumonia…")
    if not prompt and "pending_q" in st.session_state: prompt=st.session_state.pop("pending_q")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching clinical knowledge base…"): ans=get_rag_answer(prompt,store)
            st.write(ans)
        st.session_state.messages.append({"role":"assistant","content":ans})

# ── TAB 3 ──
with tab3:
    st.markdown("""<div class="about-banner"><div><div class="about-title">MediScan AI — Explainable Medical Imaging</div><div class="about-desc">A production-grade AI system for chest X-ray pneumonia detection using DenseNet-121 with Grad-CAM explainability and a RAG-powered clinical assistant grounded in WHO treatment guidelines.</div><div style="margin-top:.9rem;display:flex;gap:.4rem;flex-wrap:wrap"><span class="perf-badge">● AUC 0.96+</span><span class="perf-badge">● 90.2% Accuracy</span><span class="perf-badge">● Grad-CAM XAI</span><span class="perf-badge">● RAG + LLM</span></div></div><div class="about-stats"><div class="about-stat"><span class="about-stat-val">5,863</span><span class="about-stat-lbl">X-rays trained</span></div><div class="about-stat"><span class="about-stat-val">8</span><span class="about-stat-lbl">Epochs trained</span></div><div class="about-stat"><span class="about-stat-val">2</span><span class="about-stat-lbl">Classes detected</span></div></div></div>""",unsafe_allow_html=True)
    st.markdown(sec_label("Model Performance"),unsafe_allow_html=True)
    st.markdown(render_metric_cards(),unsafe_allow_html=True)
    st.markdown(sec_label("Technology Stack"),unsafe_allow_html=True)
    st.markdown(render_stack_table(),unsafe_allow_html=True)
    st.markdown(sec_label("Training Details"),unsafe_allow_html=True)
    d1,d2,d3=st.columns(3)
    with d1: st.markdown('<div class="info-chip" style="height:auto;padding:1.1rem 1.2rem;border-radius:12px"><div class="chip-label">Dataset</div><div class="chip-value" style="font-size:.85rem;margin-top:.3rem;color:#94a3b8">NIH Chest X-Ray 14 via Kaggle<br>5,216 train · 16 val · 624 test</div></div>',unsafe_allow_html=True)
    with d2: st.markdown('<div class="info-chip" style="height:auto;padding:1.1rem 1.2rem;border-radius:12px"><div class="chip-label">Training Config</div><div class="chip-value" style="font-size:.85rem;margin-top:.3rem;color:#94a3b8">LR: 1e-4 · Batch: 32 · Epochs: 8<br>Cosine Annealing · Adam + L2 reg</div></div>',unsafe_allow_html=True)
    with d3: st.markdown('<div class="info-chip" style="height:auto;padding:1.1rem 1.2rem;border-radius:12px"><div class="chip-label">Augmentation</div><div class="chip-value" style="font-size:.85rem;margin-top:.3rem;color:#94a3b8">Horizontal flip · Rotation ±10°<br>ColorJitter · WeightedSampler</div></div>',unsafe_allow_html=True)
    st.markdown("""<div class="built-by-card"><div class="by-avatar">👨‍💻</div><div class="by-info"><div class="by-name">B R Sathvik</div><div class="by-role">AI/ML Engineering Student · AMC Engineering College, Bengaluru · Batch 2023–2027</div></div><div class="by-links"><a class="by-link" href="https://github.com/sathvik-BR" target="_blank">↗ GitHub</a><a class="by-link" href="https://www.linkedin.com/in/b-r-sathvik-a9b785328" target="_blank">↗ LinkedIn</a></div></div>""",unsafe_allow_html=True)