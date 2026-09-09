"""Build Appendix A (open-problem ledger) from 00_INDEX.md backlog entries."""
import re, html

SRC = "/mnt/user-data/uploads/marketing_measurement/frontier_study/deep_dives/00_INDEX.md"
OUT = "/home/claude/compendium/ledger_table.tex"

# dive -> chapter label / number (chapter numbers as they appear in the book)
CH = {1:1,20:1,21:1,22:1, 2:2, 3:3,4:3, 5:4, 6:5, 7:6, 8:7,9:7, 10:8, 11:9, 12:10,
      13:11,14:11, 15:12, 16:13,19:13, 17:14, 18:15}

GREEK = {
 'α':r'\alpha','β':r'\beta','γ':r'\gamma','δ':r'\delta','ε':r'\varepsilon','ζ':r'\zeta','η':r'\eta',
 'θ':r'\theta','κ':r'\kappa','λ':r'\lambda','μ':r'\mu','ν':r'\nu','ξ':r'\xi','π':r'\pi','ρ':r'\rho',
 'σ':r'\sigma','τ':r'\tau','φ':r'\phi','χ':r'\chi','ψ':r'\psi','ω':r'\omega','Γ':r'\Gamma','Δ':r'\Delta',
 'Θ':r'\Theta','Λ':r'\Lambda','Σ':r'\Sigma','Φ':r'\Phi','Ψ':r'\Psi','Ω':r'\Omega',
}
SUBS = {'ₑ':'_e','ₜ':'_t','ⱼ':'_j','₀':'_0','₁':'_1','₂':'_2','₃':'_3','₄':'_4','₅':'_5','₉':'_9'}
SUPS = {'²':'^2','³':'^3','⁴':'^4','⁻':'^{-','⁺':'^+','ⁱ':'^i','⁰':'^0','¹':'^1'}

def tex_escape(s: str) -> str:
    # protect existing $...$ math? The INDEX uses inline $...$ sometimes; keep as is.
    out = []
    i = 0
    in_math = False
    while i < len(s):
        c = s[i]
        if c == '$':
            in_math = not in_math; out.append(c); i += 1; continue
        if c in GREEK:
            out.append(GREEK[c] if in_math else '$'+GREEK[c]+'$'); i += 1; continue
        if c in SUBS:
            t = SUBS[c]
            out.append(t if in_math else '$'+t+'$'); i += 1; continue
        if c in SUPS:
            t = SUPS[c]
            if t.endswith('{'):
                # e.g. ⁻¹ -> ^{-1}
                j = i+1; digits = ''
                while j < len(s) and s[j] in SUPS and SUPS[s[j]].startswith('^'):
                    digits += SUPS[s[j]][1:]; j += 1
                t = t + digits + '}'
                out.append(t if in_math else '$'+t+'$'); i = j; continue
            out.append(t if in_math else '$'+t+'$'); i += 1; continue
        repl = {'≈':r'$\approx$','≤':r'$\le$','≥':r'$\ge$','×':r'$\times$','→':r'$\to$','⇒':r'$\Rightarrow$',
                '−':'-','–':'--','—':'---','′':r'$^\prime$','‖':r'$\|$','·':r'$\cdot$','∝':r'$\propto$',
                '√':r'$\sqrt{}$','½':r'$\tfrac12$','¼':r'$\tfrac14$','∞':r'$\infty$','∈':r'$\in$','∖':r'$\setminus$',
                '⁄':'/','’':"'",'“':'``','”':"''",'…':r'\ldots','≠':r'$\ne$','⊥':r'$\perp$','∂':r'$\partial$',
                '⌈':r'$\lceil$','⌉':r'$\rceil$','ŝ':r'$\hat s$','ẑ':r'$\hat z$','û':r'$\hat u$','ρ̂':r'$\hat\rho$',
                'q̂':r'$\hat q$','τ̂':r'$\hat\tau$','ω̂':r'$\hat\omega$','Ĩ':r'$\tilde I$','ζ̄':r'$\bar\zeta$',
                '̂':'', '̄':'', '̃':'', 'ä':r'\"a','é':r"\'e",'ö':r'\"o','ü':r'\"u'}
        if in_math:
            m = {'≈':r'\approx','≤':r'\le','≥':r'\ge','×':r'\times','→':r'\to','⇒':r'\Rightarrow','−':'-',
                 '′':r'^\prime','‖':r'\|','·':r'\cdot','∝':r'\propto','√':r'\sqrt{}','½':r'\tfrac12','¼':r'\tfrac14',
                 '∞':r'\infty','∈':r'\in','∖':r'\setminus','⊥':r'\perp','∂':r'\partial','⌈':r'\lceil','⌉':r'\rceil'}
            if c in m: out.append(m[c]); i += 1; continue
        if c in repl:
            out.append(repl[c]); i += 1; continue
        if c in '&%#_' and not in_math:
            out.append('\\'+c); i += 1; continue
        if c == '_' and in_math:
            out.append(c); i += 1; continue
        if c == '*' :
            i += 1; continue  # markdown emphasis
        if c == '\\' and i+1 < len(s) and s[i+1] == '*':
            out.append('*'); i += 2; continue
        if c == '`':
            i += 1; continue
        out.append(c); i += 1
    return ''.join(out)

rows = []
with open(SRC, encoding='utf-8') as f:
    for line in f:
        m = re.match(r'- \*\*BL(\d+)\.\s*(.+?)\*\*\s*(.*)', line.strip())
        if not m: continue
        num = int(m.group(1)); title = m.group(2).rstrip('.')
        rest = m.group(3)
        status = 'open'
        ms = re.search(r'\*\((RESOLVED|PRICED) by dive (\d+)', rest)
        if ms: status = f"{ms.group(1).lower()} by \\dive{{{int(ms.group(2)):02d}}}"
        mf = re.search(r'\(from (\d+)', rest)
        frm = int(mf.group(1)) if mf else None
        # partial resolutions noted in queue text
        if num in (56,): status = 'partly resolved by \\dive{19}'
        if num in (57,58): status = 'resolved by \\dive{19}'
        if num in (5,6): status = 'resolved by \\dive{04}'
        if num == 7: status = 'partly advanced by \\dive{04}'
        if num == 20: status = 'resolved by \\dive{08}'
        if num == 21: status = 'resolved by \\dive{09}'
        if num == 22: status = 'partly advanced by \\dive{09}'
        if num == 42: status = 'resolved by \\dive{14}'
        if num == 82: status = 'discharged for dives 20--22 by \\dive{22}'
        if num == 75: status = 'resolved by \\dive{22}'
        ch = CH.get(frm, '')
        rows.append((num, tex_escape(title), frm, status, ch))

rows.sort()
with open(OUT, 'w') as f:
    f.write("\\begin{longtable}{@{}r p{3.35in} c p{1.55in} c@{}}\n")
    f.write("\\caption{The open-problem ledger: every backlog item (BL) spawned by the program, in numerical order, with the dive that opened it, its status as of 2026-09-07, and the chapter of this compendium in which it is discussed.}\\label{tab:ledger}\\\\\n")
    f.write("\\toprule\nBL & Problem & Opened by & Status & Chapter \\\\\n\\midrule\n\\endfirsthead\n")
    f.write("\\toprule\nBL & Problem & Opened by & Status & Chapter \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n")
    for num, title, frm, status, ch in rows:
        f.write(f"{num} & {title} & {frm:02d} & {status} & {ch} \\\\\n")
    f.write("\\end{longtable}\n")
print(len(rows), "rows written")
