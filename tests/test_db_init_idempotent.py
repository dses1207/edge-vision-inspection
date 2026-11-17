"""deploy/postgres/init 的每個檔都必須可重複執行 —— deploy/ci/migrate_db.sh 每次部署都會全部重跑。"""
import re
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "deploy" / "postgres" / "init"


def _statements(path: Path) -> list[str]:
    text = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
    return [" ".join(s.split()) for s in text.split(";") if s.strip()]


def test_every_create_and_add_column_is_guarded():
    bad = []
    for f in sorted(INIT.glob("*.sql")):
        for st in _statements(f):
            up = st.upper()
            if re.match(r"CREATE (UNIQUE )?(TABLE|INDEX|SCHEMA|EXTENSION|SEQUENCE)\b", up) and "IF NOT EXISTS" not in up:
                bad.append(f"{f.name}: {st[:80]}")
            if up.startswith("ALTER TABLE") and " ADD COLUMN " in up and "IF NOT EXISTS" not in up:
                bad.append(f"{f.name}: {st[:80]}")
            if up.startswith("CREATE DATABASE"):              # 要寫成 SELECT ... WHERE NOT EXISTS \gexec
                bad.append(f"{f.name}: {st[:80]}")
    assert not bad, "不可重複執行的語句（migrate_db.sh 每次部署都會重跑）：\n" + "\n".join(bad)
