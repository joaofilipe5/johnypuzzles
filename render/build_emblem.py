"""Build the emblem site: inject emblem particles into the template."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tpl = open(os.path.join(ROOT, "render", "emblem_template.html"), encoding="utf-8").read()
data = open(os.path.join(ROOT, "artifacts", "emblem_particles.json"), encoding="utf-8").read()
assert "/*__DATA__*/ null" in tpl
out = tpl.replace("/*__DATA__*/ null", data)
open(os.path.join(ROOT, "artifacts", "emblem_site.html"), "w", encoding="utf-8").write(out)
print(f"[flagOr] built artifacts/emblem_site.html ({len(out)} bytes)")
