from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'colab'
OUT.mkdir(parents=True, exist_ok=True)

COMMON_SETUP = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# {TITLE}\n", "Notebook Colab para **{TITLE}**.\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from pathlib import Path\n", "import os, sys\n",
        "IN_COLAB = 'google.colab' in sys.modules\n",
        "REPO_URL = 'https://github.com/GuilhermeDesoler/ai-core.git'\n",
        "REPO_DIR = Path('/content/ai-core')\n",
        "if IN_COLAB:\n",
        "    if not REPO_DIR.exists():\n",
        "        get_ipython().system(f'git clone -b improve/high-impact-training {REPO_URL} /content/ai-core')\n",
        "    get_ipython().run_line_magic('cd', '/content/ai-core')\n",
        "    get_ipython().system('pip install -q -r requirements.txt')\n",
        "else:\n",
        "    REPO_DIR = Path.cwd()\n",
        "SRC_PATH = REPO_DIR / 'src'\n",
        "if str(SRC_PATH) not in sys.path:\n",
        "    sys.path.insert(0, str(SRC_PATH))\n",
        "print('Repo:', REPO_DIR)\n", "print('SRC_PATH:', SRC_PATH)\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from pathlib import Path\n", "import shutil\n",
        "try:\n", "    from google.colab import drive\n", "    drive.mount('/content/drive')\n", "except Exception as e:\n", "    print('Drive mount opcional:', e)\n",
        "PROJECT_ROOT = REPO_DIR\n", "DATA_TARGET = PROJECT_ROOT / 'data'\n", "DATA_TARGET.mkdir(parents=True, exist_ok=True)\n",
        "SOURCE_RAW_DIR = Path('/content/drive/MyDrive/data/raw')\n", "SOURCE_PROCESSED_DIR = Path('/content/drive/MyDrive/data/processed')\n",
        "# Descomente se quiser copiar do Drive:\n",
        "# shutil.copytree(SOURCE_RAW_DIR, DATA_TARGET / 'raw', dirs_exist_ok=True)\n",
        "# shutil.copytree(SOURCE_PROCESSED_DIR, DATA_TARGET / 'processed', dirs_exist_ok=True)\n",
        "for p in [Path('data/raw/answers.json'), Path('data/processed/sequences/user_sequences.json')]:\n",
        "    print(p, p.exists())\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "# Opcional: regenere os artefatos base e sequências por sessão.\n",
        "# get_ipython().system('python run_data_pipeline.py')\n",
        "# get_ipython().system('python scripts/build_user_session_sequences.py')\n"
    ]},
]

RUN_AND_PLOT = [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import subprocess, re, json\n", "from pathlib import Path\n", "import pandas as pd\n", "import matplotlib.pyplot as plt\n",
        "SCRIPT_PATH = '{SCRIPT_PATH}'\n", "SUMMARY_NAME = '{SUMMARY_NAME}'\n",
        "cmd = f'PYTHONPATH=./src python {SCRIPT_PATH}'\n",
        "print('Running:', cmd)\n",
        "result = subprocess.run(cmd, shell=True, cwd=REPO_DIR, capture_output=True, text=True)\n",
        "print(result.stdout)\n",
        "if result.returncode != 0:\n", "    print(result.stderr)\n", "    raise RuntimeError(f'Command failed with code {result.returncode}')\n",
        "pattern = re.compile(r'Epoch\\s+(\\d+)(?:/(\\d+))?\\s*\\|\\s*Train Loss:\\s*([0-9.]+)(?:\\s*\\|\\s*Train AUC:\\s*([0-9.]+))?\\s*\\|\\s*Val Loss:\\s*([0-9.]+)\\s*\\|\\s*Val AUC:\\s*([0-9.]+)\\s*\\|\\s*Val Acc:\\s*([0-9.]+)\\s*\\|\\s*LR:\\s*([0-9.eE+-]+)')\n",
        "rows=[]\n",
        "for line in result.stdout.splitlines():\n",
        "    m = pattern.search(line)\n",
        "    if m:\n",
        "        rows.append({'epoch': int(m.group(1)), 'train_loss': float(m.group(3)), 'train_auc': float(m.group(4)) if m.group(4) else None, 'val_loss': float(m.group(5)), 'val_auc': float(m.group(6)), 'val_acc': float(m.group(7)), 'lr': float(m.group(8))})\n",
        "history_df = pd.DataFrame(rows)\n", "history_df\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fig = plt.figure(figsize=(10,5))\n", "plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')\n", "plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')\n", "plt.title(SUMMARY_NAME + ' - Loss')\n", "plt.xlabel('Epoch')\n", "plt.ylabel('Loss')\n", "plt.legend()\n", "plt.grid(alpha=0.3)\n", "plt.show()\n",
        "if history_df['train_auc'].notna().any():\n", "    fig = plt.figure(figsize=(10,5))\n", "    plt.plot(history_df['epoch'], history_df['train_auc'], label='Train AUC')\n", "    plt.plot(history_df['epoch'], history_df['val_auc'], label='Val AUC')\n", "    plt.title(SUMMARY_NAME + ' - AUC')\n", "    plt.xlabel('Epoch')\n", "    plt.ylabel('AUC')\n", "    plt.legend()\n", "    plt.grid(alpha=0.3)\n", "    plt.show()\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "saved_match = re.search(r'Saved run to\\s+(.+)', result.stdout)\n",
        "run_dir = Path(saved_match.group(1).strip()) if saved_match else None\n",
        "metrics = {}\n",
        "if run_dir and (run_dir / 'metrics.json').exists():\n", "    metrics = json.loads((run_dir / 'metrics.json').read_text())\n", "metrics\n",
        "summary_dir = Path('artifacts/colab_summaries')\n", "summary_dir.mkdir(parents=True, exist_ok=True)\n",
        "(summary_dir / f'{SUMMARY_NAME}_history.csv').write_text(history_df.to_csv(index=False))\n",
        "(summary_dir / f'{SUMMARY_NAME}_metrics.json').write_text(json.dumps(metrics, indent=2))\n",
        "print('Saved summary files to', summary_dir)\n"
    ]},
]

TASKS = [
    ('01_dkt_next_item_colab.ipynb', 'DKT Next-Item', 'scripts/train_eval_dkt_next_item.py', 'dkt_next_item'),
    ('02_dkt_h2_colab.ipynb', 'DKT H2', 'scripts/train_eval_dkt_topic_h2.py', 'dkt_h2'),
    ('03_dkt_h3_colab.ipynb', 'DKT H3', 'scripts/train_eval_dkt_topic_h3.py', 'dkt_h3'),
    ('04_lpkt_next_item_colab.ipynb', 'LPKT Next-Item', 'scripts/train_eval_lpkt_next_item.py', 'lpkt_next_item'),
    ('05_lpkt_h2_colab.ipynb', 'LPKT H2', 'scripts/train_eval_lpkt_topic_h2.py', 'lpkt_h2'),
    ('06_lpkt_h3_colab.ipynb', 'LPKT H3', 'scripts/train_eval_lpkt_topic_h3.py', 'lpkt_h3'),
    ('07_dkvmn_next_item_colab.ipynb', 'DKVMN Next-Item', 'scripts/train_eval_dkvmn_next_item.py', 'dkvmn_next_item'),
    ('08_dkvmn_h2_colab.ipynb', 'DKVMN H2', 'scripts/train_eval_dkvmn_topic_h2.py', 'dkvmn_h2'),
    ('09_dkvmn_h3_colab.ipynb', 'DKVMN H3', 'scripts/train_eval_dkvmn_topic_h3.py', 'dkvmn_h3'),
]


def replace(obj, title: str, script: str, summary: str):
    if isinstance(obj, str):
        return obj.replace('{TITLE}', title).replace('{SCRIPT_PATH}', script).replace('{SUMMARY_NAME}', summary)
    if isinstance(obj, list):
        return [replace(x, title, script, summary) for x in obj]
    if isinstance(obj, dict):
        return {k: replace(v, title, script, summary) for k, v in obj.items()}
    return obj

for filename, title, script, summary in TASKS:
    cells = replace(COMMON_SETUP + RUN_AND_PLOT, title, script, summary)
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    (OUT / filename).write_text(json.dumps(nb, ensure_ascii=False))

compare_nb = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# Compare all KT runs\n', 'Lê os summaries salvos pelos notebooks e compara métricas dos 9 experimentos.\n']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "from pathlib import Path\n", "import json, sys\n", "import pandas as pd\n", "import matplotlib.pyplot as plt\n",
            "IN_COLAB = 'google.colab' in sys.modules\n", "REPO_URL = 'https://github.com/GuilhermeDesoler/ai-core.git'\n", "REPO_DIR = Path('/content/ai-core')\n",
            "if IN_COLAB:\n", "    if not REPO_DIR.exists():\n", "        get_ipython().system(f'git clone -b improve/high-impact-training {REPO_URL} /content/ai-core')\n", "    get_ipython().run_line_magic('cd', '/content/ai-core')\n", "    get_ipython().system('pip install -q -r requirements.txt')\n", "else:\n", "    REPO_DIR = Path.cwd()\n",
            "summary_dir = REPO_DIR / 'artifacts' / 'colab_summaries'\n", "print('Summary dir:', summary_dir)\n", "print('Exists:', summary_dir.exists())\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "rows=[]\n", "for p in sorted(summary_dir.glob('*_metrics.json')):\n", "    data=json.loads(p.read_text()) if p.exists() else {}\n", "    name=p.stem.replace('_metrics','')\n", "    model, task = name.split('_',1)\n", "    rows.append({'name':name,'model':model.upper(),'task':task,'best_val_auc':data.get('best_val_auc'),'best_val_loss':data.get('best_val_loss'),'test_auc':data.get('test_auc'),'test_loss':data.get('test_loss'),'test_acc':data.get('test_acc')})\n", "results_df=pd.DataFrame(rows).sort_values(['task','model'])\n", "results_df\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "for metric in ['test_auc','test_loss','test_acc']:\n", "    pivot=results_df.pivot(index='task',columns='model',values=metric)\n", "    display(pivot)\n", "    fig=plt.figure(figsize=(10,5))\n", "    pivot.plot(kind='bar', ax=plt.gca())\n", "    plt.title(f'Comparison - {metric}')\n", "    plt.ylabel(metric)\n", "    plt.grid(alpha=0.3)\n", "    plt.show()\n"
        ]},
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}},
    'nbformat': 4,
    'nbformat_minor': 5,
}
(OUT / '10_compare_all_models_colab.ipynb').write_text(json.dumps(compare_nb, ensure_ascii=False))

eda_nb = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['# EDA full analysis for KT\n', 'Análise exploratória forte focada em sequência, tempo, sessões e adequação para KT.\n']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "from pathlib import Path\n", "import json, sys\n", "import pandas as pd\n", "import numpy as np\n", "import matplotlib.pyplot as plt\n",
            "IN_COLAB='google.colab' in sys.modules\n", "REPO_URL='https://github.com/GuilhermeDesoler/ai-core.git'\n", "REPO_DIR=Path('/content/ai-core')\n",
            "if IN_COLAB:\n", "    if not REPO_DIR.exists():\n", "        get_ipython().system(f'git clone -b improve/high-impact-training {REPO_URL} /content/ai-core')\n", "    get_ipython().run_line_magic('cd','/content/ai-core')\n", "    get_ipython().system('pip install -q -r requirements.txt')\n", "else:\n", "    REPO_DIR=Path.cwd()\n",
            "answers=pd.read_csv(REPO_DIR/'data/processed/dataset/answers_prepared.csv')\n", "with open(REPO_DIR/'data/processed/sequences/user_sequences.json') as f:\n", "    sequences=json.load(f)\n", "answers['timestamp_dt']=pd.to_datetime(answers['timestamp'], unit='ms', errors='coerce')\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "overview=pd.Series({'rows':len(answers),'users':answers['user_id'].nunique(),'questions':answers['question_id'].nunique(),'skills':answers['skill_id'].nunique(),'correct_rate':float(answers['correct'].mean()),'dup_rows':int(answers.duplicated().sum()),'time_zero_rate':float((answers['time_response']==0).mean())})\n", "overview.to_frame('value')\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "user_stats=answers.groupby('user_id').agg(n_interactions=('question_id','size'),n_skills=('skill_id','nunique'),first_ts=('timestamp_dt','min'),last_ts=('timestamp_dt','max')).reset_index()\n", "user_stats['timespan_hours']=((user_stats['last_ts']-user_stats['first_ts']).dt.total_seconds()/3600).fillna(0)\n", "display(user_stats[['n_interactions','n_skills','timespan_hours']].describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]))\n", "fig=plt.figure(figsize=(10,5)); plt.hist(user_stats['n_interactions'], bins=50); plt.title('Interactions per user'); plt.show()\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "answers=answers.sort_values(['user_id','timestamp']).copy()\n", "answers['delta_t_ms']=answers.groupby('user_id')['timestamp'].diff().fillna(0)\n", "answers['delta_t_min']=answers['delta_t_ms']/60000\n", "display(pd.Series({'delta_p50_min':float(answers['delta_t_min'].quantile(0.5)),'delta_p90_min':float(answers['delta_t_min'].quantile(0.9)),'delta_p99_min':float(answers['delta_t_min'].quantile(0.99)),'time_p99_sec':float(answers['time_response'].quantile(0.99)/1000)}).to_frame('value'))\n", "fig=plt.figure(figsize=(10,5)); plt.hist(np.log1p(answers['delta_t_ms']), bins=60); plt.title('log1p(delta_t_ms)'); plt.show()\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "SESSION_GAP_MIN=60\n", "answers['new_session_flag']=((answers['delta_t_min']>SESSION_GAP_MIN)|(answers.groupby('user_id').cumcount()==0)).astype(int)\n", "answers['session_idx']=answers.groupby('user_id')['new_session_flag'].cumsum()\n", "session_stats=answers.groupby(['user_id','session_idx']).agg(n_interactions=('question_id','size'),n_skills=('skill_id','nunique')).reset_index()\n", "display(session_stats[['n_interactions','n_skills']].describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]))\n", "fig=plt.figure(figsize=(10,5)); plt.hist(session_stats['n_interactions'], bins=40); plt.title('Session length'); plt.show()\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "answers['prev_skill_id']=answers.groupby('user_id')['skill_id'].shift(1)\n", "answers['skill_switch']=((answers['prev_skill_id'].notna())&(answers['skill_id']!=answers['prev_skill_id'])).astype(int)\n", "frag=answers.groupby('user_id').agg(n_interactions=('question_id','size'),n_skills=('skill_id','nunique'),switches=('skill_switch','sum')).reset_index()\n", "frag['switch_rate']=frag['switches']/(frag['n_interactions']-1).clip(lower=1)\n", "frag['interactions_per_skill']=frag['n_interactions']/frag['n_skills'].clip(lower=1)\n", "display(frag[['switch_rate','interactions_per_skill']].describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]))\n", "fig=plt.figure(figsize=(10,5)); plt.hist(frag['switch_rate'], bins=40); plt.title('Skill switch rate'); plt.show()\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "user_skill_counts=answers.groupby(['user_id','skill_id']).size().reset_index(name='n_attempts')\n", "repeat_summary=pd.Series({'mean_attempts_per_user_skill':float(user_skill_counts['n_attempts'].mean()),'median_attempts_per_user_skill':float(user_skill_counts['n_attempts'].median()),'share_attempted_once':float((user_skill_counts['n_attempts']==1).mean()),'share_attempted_ge_3':float((user_skill_counts['n_attempts']>=3).mean())})\n", "repeat_summary.to_frame('value')\n"
        ]},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "diagnostic=pd.DataFrame([{'criterion':'long user histories','value':float(user_stats['n_interactions'].median()),'comment':'higher is better for KT'},{'criterion':'session length','value':float(session_stats['n_interactions'].median()),'comment':'short sessions hurt temporal KT'},{'criterion':'skill switch rate','value':float(frag['switch_rate'].median()),'comment':'high values mean fragmented study'},{'criterion':'repeat per user-skill >=3','value':float((user_skill_counts['n_attempts']>=3).mean()),'comment':'higher helps mastery modeling'},{'criterion':'time_response zero rate','value':float((answers['time_response']==0).mean()),'comment':'high values reduce temporal signal'}])\n", "diagnostic\n"
        ]},
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}},
    'nbformat': 4,
    'nbformat_minor': 5,
}
(OUT / '11_eda_full_analysis_colab.ipynb').write_text(json.dumps(eda_nb, ensure_ascii=False))

print('Generated notebooks in', OUT)
for p in sorted(OUT.glob('*.ipynb')):
    print('-', p.name)
