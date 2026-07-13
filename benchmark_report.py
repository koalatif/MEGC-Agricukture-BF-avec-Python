import pandas as pd
import re
from pathlib import Path
import sys

def parse_results(text: str) -> pd.DataFrame:
    # Match blocks starting with T\d+
    blocks = re.split(r'(?=T\d+\s*-)', text)
    rows = []
    for block in blocks:
        block = block.strip()
        if not block: continue
        m_title = re.search(r'(T\d+)\s*-\s*(.*)', block.split('\n')[0])
        if not m_title: continue
        test_id = m_title.group(1)
        desc = m_title.group(2).strip()
        
        m_res = re.search(r'res=([\d.eE+-]+)', block)
        m_walras = re.search(r'walras=([\d.eE+-]+)', block)
        m_outcome = re.search(r'->\s*(REUSSI|ECHEC)', block)
        
        rows.append({
            "Test": test_id,
            "Description": desc,
            "Residual": float(m_res.group(1)) if m_res else None,
            "Walras": float(m_walras.group(1)) if m_walras else None,
            "Outcome": m_outcome.group(1) if m_outcome else "UNKNOWN"
        })
    return pd.DataFrame(rows)

def main():
    if len(sys.argv) > 1:
        try:
            text = Path(sys.argv[1]).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = Path(sys.argv[1]).read_text(encoding='utf-16')
    else:
        text = sys.stdin.read()
    
    if not text.strip():
        print("No input text provided.")
        return
        
    df = parse_results(text)
    md_table = df.to_markdown(index=False)
    print(md_table)
    Path("benchmark_summary.md").write_text(md_table, encoding='utf-8')
    print("\nSaved to benchmark_summary.md")

if __name__ == "__main__":
    main()
