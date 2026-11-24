const number = new Intl.NumberFormat("en")
const node = (tag, text, className) => {
  const el = document.createElement(tag)
  if (text !== undefined) el.textContent = text
  if (className) el.className = className
  return el
}

const styles = `
.hash-ruler{--hr-ink:#1b2621;--hr-green:#d3fa69;--hr-orange:#a63e1e;display:grid;gap:1.1rem;color:var(--hr-ink);min-width:0}
.hash-ruler p,.hash-ruler figure{margin:0}.hr-choices,.hr-probes{display:flex;gap:.45rem;flex-wrap:wrap}
.hash-ruler button{font:inherit;cursor:pointer;padding:.42rem .85rem;color:var(--hr-ink);border:1px solid #a1ab9d;border-radius:2rem;background:transparent;line-height:1.25}
.hash-ruler button:hover{background:#e8eddf}.hash-ruler button[aria-pressed=true]{background:var(--hr-ink);border-color:var(--hr-ink);color:white}
.hash-ruler .hr-field{background:#1b2621;color:#f8f6ef;padding:clamp(1rem,3vw,2rem);border-radius:.9rem}
.hr-readout{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;flex-wrap:wrap}.hr-readout strong{font:600 clamp(1.3rem,3vw,2rem)/1.2 var(--sans,system-ui)}
.hr-readout code,.hr-axis{font:500 .8rem/1.45 var(--mono,monospace);color:#d3fa69}
.hr-field canvas{display:block;width:100%;height:125px;cursor:crosshair;margin-top:.6rem}
.hr-axis{display:flex;justify-content:space-between;color:#cbd7cd}.hr-slider{width:100%;margin:.55rem 0 .35rem;accent-color:var(--hr-green);cursor:pointer}
.hr-legend{font-size:.85rem;color:#e9eede}.hr-legend b{color:#d3fa69}.hr-legend em{color:#ffb49e;font-style:normal}.hr-probes{align-items:center;margin-top:.4rem}.hr-probes span{margin-right:.3rem}
.hr-summary{display:grid;grid-template-columns:1fr 1fr;gap:clamp(1rem,3vw,2.5rem)}.hr-summary>div{border-top:1px solid #b5b9ac;padding-top:.7rem}
.hr-summary h3{margin:0 0 .35rem;font:650 1.14rem/1.3 var(--sans,system-ui)}.hr-summary p{font-size:.96rem;line-height:1.5}
.hr-songs{list-style:none;margin:.55rem 0 0;padding:0;display:flex;gap:.35rem .75rem;flex-wrap:wrap}
.hr-songs li{font-size:.86rem;border-left:2px solid #a5b685;padding-left:.45rem}
@media(max-width:640px){.hr-summary{grid-template-columns:1fr}.hr-field canvas{height:100px}.hr-choices button{font-size:.9rem}}
`

export async function mountRuler(container) {
  const resp = await fetch("/ruler.json")
  if (!resp.ok) throw Error("The song map could not load.")
  const data = await resp.json()
  container.replaceChildren()
  container.classList.add("hash-ruler")
  const css = node("style", styles)
  const intro = node(
    "p",
    "Choose a clip to follow. Its fingerprint is a pattern made from the sound. Ten yes-or-no checks turn that pattern into a short code called a hash.",
  )
  const choices = node("div", undefined, "hr-choices")
  choices.setAttribute("role", "group")
  choices.setAttribute("aria-label", "Pick a song example")
  const field = node("figure", undefined, "hr-field")
  const readout = node("figcaption", undefined, "hr-readout")
  const bucketName = node("strong"),
    code = node("code")
  readout.append(bucketName, code)
  const canvas = node("canvas")
  canvas.setAttribute(
    "aria-label",
    "Heights show the number of clips in each box. Use the slider below to explore.",
  )
  const axis = node("div", undefined, "hr-axis")
  axis.setAttribute("aria-hidden", "true")
  for (const mark of [0, 256, 512, 768, 1023]) axis.append(node("span", String(mark)))
  const slider = node("input", undefined, "hr-slider")
  Object.assign(slider, { type: "range", min: "0", max: "1023", step: "1" })
  slider.setAttribute("aria-label", "Explore a box numbered 0 to 1023")
  const legend = node("p", undefined, "hr-legend")
  legend.append(
    "Taller marks hold more clips. ",
    node("b", "Green: the clip’s box."),
    " ",
    node("em", "Orange: three extra boxes to check."),
  )
  field.append(readout, canvas, axis, slider, legend)
  const guide = node(
    "p",
    "The box numbers are addresses. Flipping just one yes-or-no answer can send a clip far along the ruler. The search tries a few extra boxes in case noise changed an answer.",
  )
  const probes = node("div", undefined, "hr-probes")
  probes.setAttribute("role", "group")
  probes.setAttribute("aria-label", "The four boxes checked for this clip")
  const summary = node("div", undefined, "hr-summary")
  const contents = node("div"),
    result = node("div")
  const bucketHeading = node("h3"),
    bucketText = node("p"),
    songs = node("ul", undefined, "hr-songs")
  contents.append(bucketHeading, bucketText, songs)
  result.append(node("h3", "Which song wins?"))
  const verdict = node("p")
  result.append(verdict)
  summary.append(contents, result)
  container.append(css, intro, choices, field, guide, probes, summary)
  const largest = Math.max(...data.counts)
  const songTitle = (id) => data.labels[id][0]
  let example, selected
  function draw() {
    const w = canvas.clientWidth,
      h = canvas.clientHeight
    if (!w || !h) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    const ctx = canvas.getContext("2d")
    ctx.scale(dpr, dpr)
    const barWidth = w / data.counts.length
    ctx.fillStyle = "#a7b5a8"
    for (let i = 0; i < data.counts.length; i++) {
      const height = ((h - 12) * data.counts[i]) / largest
      ctx.fillRect(i * barWidth, h - height, Math.max(0.75, barWidth), height)
    }
    example.probes.forEach((bucket, i) => {
      const x = (bucket + 0.5) * barWidth
      ctx.fillStyle = i === 0 ? "#d3fa69" : "#ffad8a"
      ctx.fillRect(x - 1, h * 0.12, 2, h * 0.88)
    })
    const x = (selected + 0.5) * barWidth
    ctx.fillStyle = "#fff"
    ctx.beginPath()
    ctx.moveTo(x - 5, 0)
    ctx.lineTo(x + 5, 0)
    ctx.lineTo(x, 8)
    ctx.fill()
    ctx.setLineDash([2, 3])
    ctx.strokeStyle = "#fff"
    ctx.beginPath()
    ctx.moveTo(x, 9)
    ctx.lineTo(x, h)
    ctx.stroke()
  }
  function inspect(bucket) {
    selected = bucket
    slider.value = String(bucket)
    bucketName.textContent = "Box " + bucket
    code.textContent = "Hash " + bucket.toString(2).padStart(10, "0")
    bucketHeading.textContent = "Inside box " + bucket
    const count = data.counts[bucket]
    bucketText.textContent =
      number.format(count) +
      " saved " +
      (count === 1 ? "clip" : "clips") +
      (count ? ". Songs here include:" : ".")
    slider.setAttribute("aria-valuetext", "Box " + bucket + ", " + count + " clips")
    songs.replaceChildren(...data.examples[bucket].map((id) => node("li", songTitle(id))))
    ;[...probes.querySelectorAll("button")].forEach((b) =>
      b.setAttribute("aria-pressed", String(Number(b.value) === bucket)),
    )
    draw()
  }
  function selectCase(index) {
    example = data.cases[index]
    ;[...choices.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)))
    probes.replaceChildren(node("span", "Jump to a searched box"))
    for (const [i, k] of example.probes.entries()) {
      const b = node("button", String(k))
      b.type = "button"
      b.value = String(k)
      b.setAttribute("aria-label", (i ? "Extra box " : "Clip’s box ") + k)
      b.onclick = () => inspect(k)
      probes.append(b)
    }
    const winning = songTitle(example.winner)
    verdict.textContent =
      "This is one of 64 maps. Together they picked " +
      number.format(example.candidates) +
      " of " +
      number.format(data.windows) +
      " clips for a closer listen. " +
      winning +
      " came closest. " +
      (example.accepted ? "The search named the song." : "The match was too uncertain to name a song.")
    inspect(example.probes[0])
  }
  data.cases.forEach((c, i) => {
    const b = node("button", songTitle(c.source))
    b.type = "button"
    b.onclick = () => selectCase(i)
    choices.append(b)
  })
  slider.oninput = () => inspect(Number(slider.value))
  canvas.onclick = (e) => {
    const box = canvas.getBoundingClientRect()
    inspect(Math.min(1023, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * 1024))))
  }
  selectCase(0)
  new ResizeObserver(draw).observe(canvas)
}
