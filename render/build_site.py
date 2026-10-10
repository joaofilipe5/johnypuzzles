"""Build the flagOr site: inject particle data into the template -> artifacts/flagor_site.html."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tpl = open(os.path.join(ROOT, "render", "site_template.html"), encoding="utf-8").read()
data = open(os.path.join(ROOT, "artifacts", "flag_particles.json"), encoding="utf-8").read()
assert "/*__DATA__*/ null" in tpl, "placeholder missing in template"
out = tpl.replace("/*__DATA__*/ null", data)
dst = os.path.join(ROOT, "artifacts", "flagor_site.html")
open(dst, "w", encoding="utf-8").write(out)
print(f"[flagOr] built {dst}  ({len(out)} bytes, {data.count('[', 1)} particles)")
