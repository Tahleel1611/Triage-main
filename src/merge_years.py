import pandas as pd
import re
import os

def parse_nhamcs_line(line):
    """
    Parses a single line of the NHAMCS ED fixed-width file.
    Offsets are generally consistent for 2021-2022.
    """
    try:
        # Age (Approx index 10-14)
        age_str = line[10:14]
        
        # Vitals Block (Approx index 47-66)
        temp_str = line[47:51]
        pulse_str = line[51:54]
        resp_str = line[54:57]
        sbp_str = line[57:60]
        dbp_str = line[60:63]
        o2_str = line[63:66]
        
        # ESI Triage Level (Target) - Index 66-68
        esi_str = line[66:68]
        
        # New Features (Discovered via exploration)
        # Arrival Mode (1=EMS, 2=Other, 3=Unknown) - Index 30:31
        arrival_str = line[30:31]
        
        # Pain Scale (0-10) - Index 174:176
        pain_str = line[174:176]

        # Diagnosis/Reason Codes
        rest_of_line = line[68:]
        diags = re.findall(r'[A-Z]\d{2,4}', rest_of_line)
        diag_text = " ".join(diags)

        def clean_num(s, is_temp=False):
            if not s: return None
            s = s.strip()
            if not s or s.startswith('-') or not s.isdigit():
                return None
            val = float(s)
            if is_temp:
                return val / 10.0 
            return val

        return {
            'Age': clean_num(age_str),
            'Temp': clean_num(temp_str, is_temp=True),
            'Pulse': clean_num(pulse_str),
            'Resp': clean_num(resp_str),
            'SBP': clean_num(sbp_str),
            'DBP': clean_num(dbp_str),
            'O2Sat': clean_num(o2_str),
            'ESI': clean_num(esi_str),
            'ArrivalMode': clean_num(arrival_str),
            'PainScale': clean_num(pain_str),
            'Chief_complain': diag_text if diag_text else "Unknown"
        }
    except Exception:
        return None

def process_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parsed = parse_nhamcs_line(line)
            if parsed:
                data.append(parsed)
    return pd.DataFrame(data)

def main():
    # 1. Setup Paths
    files = {
        '2022': 'data/ed2022/ed2022',
        '2021': 'data/ed2021/ed2021'
    }

    # 2. Parse
    all_dfs = []
    
    for year, path in files.items():
        if os.path.exists(path):
            print(f"Processing {year} data from {path}...")
            df = process_file(path)
            df['Year'] = int(year)
            all_dfs.append(df)
        else:
            print(f"Warning: File not found for {year} at {path}")

    # 3. Merge
    if not all_dfs:
        print("No data found!")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Filter
    final_df = final_df.dropna(subset=['ESI'])
    final_df = final_df[final_df['ESI'].between(1, 5)]
    
    print(f"Total Combined Rows: {len(final_df)}")
    print(final_df['Year'].value_counts())
    
    output_path = 'data/nhamcs_combined.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")

if __name__ == "__main__":
    main()
