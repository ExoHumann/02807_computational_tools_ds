const fmt = (n) => n.toLocaleString("en")
const binary = (value) => value.toString(2).padStart(10, "0")
const seconds = (n) => (Number.isInteger(n) ? String(n) : n.toFixed(2))
const stages = ["The sound", "Ten answers", "Inside the boxes", "The song timeline"]
const changes = {
  noise: "with background noise",
  "time_stretch+pitch_shift": "with its speed and pitch changed",
}
const cache = new Map()
const el = (tag, text, cls) => {
  const n = document.createElement(tag)
  if (text !== undefined) n.textContent = text
  if (cls) n.className = cls
  return n
}
function button(text, action, cls) {
  const b = el("button", text, cls)
  b.type = "button"
  b.onclick = action
  return b
}
function heading(title, explanation) {
  const wrap = el("div", undefined, "fp-heading")
  wrap.append(el("h3", title), el("p", explanation))
  return wrap
}
function group(label, cls) {
  const wrap = el("div", undefined, cls)
  wrap.setAttribute("role", "group")
  wrap.setAttribute("aria-label", label)
  return wrap
}
function mark(buttons, index) {
  ;[...buttons.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)))
}
function loadBoxes(part) {
  if (!cache.has(part)) {
    const request = fetch("/ruler-boxes/" + part + ".json")
      .then((r) => {
        if (!r.ok) throw Error("Missing box data")
        return r.json()
      })
      .catch((error) => {
        cache.delete(part)
        throw error
      })
    cache.set(part, request)
  }
  return cache.get(part)
}
function bars(values, cls, caption) {
  const graph = el("div", undefined, cls)
  graph.setAttribute("role", "img")
  graph.setAttribute("aria-label", caption)
  for (const v of values) {
    const bar = el("i")
    bar.style.setProperty("--height", Math.abs(v) * 100 + "%")
    if (v < 0) bar.className = "negative"
    graph.append(bar)
  }
  return graph
}
function soundPanel(example) {
  const track = example.timelines[example.source]
  const panel = el("div")
  panel.append(
    heading(
      "Turn sound into a fingerprint.",
      "We gave the search ten seconds of “" +
        track.title +
        "” " +
        changes[example.change] +
        ". Here’s where the clip came from and what the search kept.",
    ),
  )
  const pair = el("div", undefined, "fp-sound-pair"),
    waveCard = el("figure", undefined, "fp-sound-card"),
    printCard = el("figure", undefined, "fp-print-card")
  const wave = bars(
    example.waveform,
    "fp-wave",
    "Sound level along the saved recording, with the test clip’s starting point marked.",
  )
  const marker = el("span", undefined, "fp-marker")
  marker.style.left = (100 * example.start) / track.duration + "%"
  wave.append(marker)
  const axis = el("div", undefined, "fp-axis")
  axis.append(el("span", "0 s"), el("span", seconds(track.duration) + " s"))
  waveCard.append(
    el("h4", "The original recording"),
    wave,
    axis,
    el(
      "figcaption",
      "The test clip starts at " + seconds(example.start) + " s. Taller marks mean louder sound.",
    ),
  )
  const toggle = group("Explore the saved numbers", "fp-feature-toggle"),
    numbers = el("div"),
    caption = el("figcaption")
  function display(i) {
    mark(toggle, i)
    const vector = i ? example.fingerprint : example.features
    const scale = Math.max(...vector.map(Math.abs))
    numbers.replaceChildren(
      bars(
        vector.map((v) => v / scale),
        "fp-pattern",
        "All " + vector.length + " values in order, shown relative to the largest size in this view.",
      ),
    )
    caption.textContent = i
      ? "A trained layer turns the model’s description into these 256 numbers. This is the fingerprint that goes into the ten checks."
      : "A sound model listens to the altered clip and describes it with 1,024 numbers. Each mark is one number in that description."
  }
  toggle.append(
    button("Sound model · 1,024", () => display(0)),
    button("Fingerprint · 256", () => display(1)),
  )
  printCard.append(el("h4", "What the computer remembers"), toggle, numbers, caption)
  display(1)
  pair.append(waveCard, printCard)
  panel.append(pair)
  return panel
}

function checksPanel(example, data, step, advance, openBox) {
  const panel = el("div")
  panel.append(
    heading(
      "Build a box address.",
      "On this map, ten checks sort the clips into boxes. Each check mixes the fingerprint numbers in a different way. If the total is zero or above, the answer is yes (1). Below zero means no (0).",
    ),
  )
  const code = group("Choose a check", "fp-bits")
  const result = el("div", undefined, "fp-check-result")
  const address = binary(example.probes[0])
  const checks = [...address].map((v, i) => {
    const b = button("", () => show(i), "fp-bit")
    b.setAttribute("aria-label", "Check " + (i + 1))
    b.append(el("small", String(i + 1)), el("strong", v))
    code.append(b)
    return b
  })
  const barsBox = el("div", undefined, "fp-split")
  const text = el("p", undefined, "fp-branch-note")
  const why = el("details", undefined, "fp-math"),
    math = el("p")
  why.append(el("summary", "How did this check decide?"), math)
  const controls = el("div", undefined, "fp-inline-actions")
  const prev = button("← Previous check", () => show(step - 1), "fp-quiet")
  const next = button(
    "Next check →",
    () => {
      if (step === 9) openBox(example.probes[0])
      else show(step + 1)
    },
    "fp-primary",
  )
  controls.append(prev, next)
  function show(i) {
    step = i
    advance(i)
    mark(code, i)
    checks.forEach((b, idx) => {
      b.classList.toggle("unread", idx > i)
      b.lastChild.textContent = idx > i ? "·" : address[idx]
    })
    const prefix = address.slice(0, i),
      bit = address[i],
      start = parseInt((prefix + "0").padEnd(10, "0"), 2)
    const width = 2 ** (9 - i)
    const sum = (first) => data.counts.slice(first, first + width).reduce((a, b) => a + b, 0)
    const branches = [sum(start), sum(start + width)],
      value = example.projections[9 - i]
    const label = el("strong", "Check " + (i + 1) + " says " + (bit === "1" ? "yes" : "no") + ".")
    const number = Math.abs(value) < 0.0005 ? value.toExponential(2) : value.toFixed(3)
    result.replaceChildren(label, el("span", "Its total is " + number + "."))
    const [yes, no] = example.pulls[9 - i]
    math.textContent =
      "Each of the 256 numbers gets a weight for this check. Added together, the parts pulling toward yes reached " +
      yes.toFixed(4) +
      ". The parts pulling toward no reached " +
      no.toFixed(4) +
      ". " +
      (bit === "1" ? "Yes" : "No") +
      " had the stronger pull."
    barsBox.replaceChildren(
      ...branches.map((n, index) => {
        const branch = el("div", undefined, "fp-branch" + (String(index) === bit ? " follows" : ""))
        const header = el("div")
        header.append(el("b", index ? "Yes" : "No"), el("span", fmt(n) + " clips"))
        const rail = el("div", undefined, "fp-rail"),
          fill = el("i")
        fill.style.width = (100 * n) / (branches[0] + branches[1]) + "%"
        rail.append(fill)
        branch.append(header, rail)
        return branch
      }),
    )
    text.textContent =
      i === 9
        ? "All ten answers lead to box " +
          example.probes[0] +
          ", which holds " +
          fmt(branches[Number(bit)]) +
          " clips."
        : fmt(branches[Number(bit)]) +
          " of " +
          fmt(branches[0] + branches[1]) +
          " saved clips share the answers so far on this map."
    prev.disabled = i === 0
    next.textContent = i === 9 ? "Open box " + example.probes[0] + " →" : "Next check →"
  }
  panel.append(code, result, barsBox, text, why, controls)
  show(step)
  return panel
}

function boxBrowser(example, data, initial, remember, onDispose) {
  const panel = el("div")
  panel.append(
    heading(
      "Open a box.",
      "The search opens the clip’s own box and three more. Those extra boxes try the other answers to the two checks closest to zero.",
    ),
  )
  const probes = group("The four searched boxes", "fp-probes")
  const field = el("div", undefined, "fp-map"),
    top = el("div", undefined, "fp-map-top"),
    boxName = el("strong"),
    address = el("code")
  top.append(boxName, address)
  const canvas = el("canvas")
  canvas.setAttribute(
    "aria-label",
    "All 1,024 boxes on this map. Taller marks mean more clips. Use the slider to explore.",
  )
  const axis = el("div", undefined, "fp-axis")
  axis.append(el("span", "0"), el("span", "256"), el("span", "512"), el("span", "768"), el("span", "1,023"))
  const slider = el("input", undefined, "hr-slider")
  Object.assign(slider, { type: "range", min: 0, max: 1023, step: 1 })
  slider.setAttribute("aria-label", "Explore a box")
  field.append(
    top,
    canvas,
    axis,
    slider,
    el("p", "Taller marks hold more clips. The four tall lines show the boxes we checked."),
  )
  const inside = el("div", undefined, "fp-box-content"),
    insideHead = el("h4"),
    explanation = el("p"),
    tiles = group("Songs in this box", "fp-tiles"),
    detail = el("div", undefined, "fp-song-detail")
  const picker = el("select")
  picker.setAttribute("aria-label", "Find a song in this box")
  inside.append(insideHead, explanation, tiles, picker, detail)
  let selected = initial,
    shown = [],
    version = 0
  const peak = Math.max(...data.counts)
  function draw() {
    const w = canvas.clientWidth,
      h = canvas.clientHeight
    if (!w || !h) return
    const d = Math.min(devicePixelRatio || 1, 2)
    canvas.width = Math.round(w * d)
    canvas.height = Math.round(h * d)
    const c = canvas.getContext("2d")
    c.scale(d, d)
    data.counts.forEach((n, i) => {
      c.fillStyle = "#b8c8b5"
      const v = ((h - 10) * n) / peak
      c.fillRect((i * w) / 1024, h - v, Math.max(0.7, w / 1024), v)
    })
    example.probes.forEach((b, i) => {
      c.fillStyle = i ? "#ffac86" : "#d3fa69"
      c.fillRect(((b + 0.5) * w) / 1024 - 1, h * 0.14, 2, h * 0.86)
    })
    const x = ((selected + 0.5) * w) / 1024
    c.fillStyle = "#fff"
    c.beginPath()
    c.moveTo(x - 5, 0)
    c.lineTo(x + 5, 0)
    c.lineTo(x, 7)
    c.fill()
    c.setLineDash([2, 3])
    c.strokeStyle = "white"
    c.beginPath()
    c.moveTo(x, 7)
    c.lineTo(x, h)
    c.stroke()
  }
  function selectSong(i) {
    picker.selectedIndex = i
    const tile = tiles.children[i]
    if (tile.offsetTop < tiles.scrollTop) tiles.scrollTop = tile.offsetTop
    else if (tile.offsetTop + tile.offsetHeight > tiles.scrollTop + tiles.clientHeight)
      tiles.scrollTop = tile.offsetTop + tile.offsetHeight - tiles.clientHeight
    ;[...tiles.children].forEach((b, j) => {
      b.setAttribute("aria-pressed", String(j === i))
      b.tabIndex = j === i ? 0 : -1
    })
    const [, title, artist, duration, ...starts] = shown[i]
    const clips = el("div", undefined, "fp-times")
    starts.forEach((s) => clips.append(el("span", s + "–" + (s + data.windowSeconds) + " s")))
    const strip = el("div", undefined, "fp-mini-map")
    strip.setAttribute("role", "img")
    strip.setAttribute(
      "aria-label",
      "Where the pieces in this box sit in the " + seconds(duration) + "-second recording",
    )
    for (const start of starts) {
      const lane = el("div"),
        mark = el("i")
      mark.style.left = (100 * start) / duration + "%"
      mark.style.width = (100 * data.windowSeconds) / duration + "%"
      lane.append(mark)
      strip.append(lane)
    }
    const axis = el("div", undefined, "fp-axis")
    axis.append(el("span", "0 s"), el("span", seconds(duration) + " s"))
    detail.replaceChildren(el("strong", title), el("span", artist), strip, axis, clips)
    if (shown[i][0] === example.source)
      detail.append(el("small", "This is the song our test clip came from."))
  }
  async function inspect(box) {
    selected = box
    remember(box)
    const myVersion = ++version
    slider.value = box
    draw()
    const home = example.probes[0],
      probe = example.probes.indexOf(box)
    const changed = [...binary(home ^ box)].flatMap((v, i) => (v === "1" ? [i + 1] : []))
    const status =
      probe === 0
        ? "Clip’s own box"
        : probe > 0
          ? "Extra box · changed " + (changed.length === 1 ? "check " : "checks ") + changed.join(" and ")
          : "Outside the four searched boxes"
    boxName.textContent = "Box " + box
    address.textContent = binary(box) + " · " + status
    slider.setAttribute("aria-valuetext", "Box " + box + ", " + data.counts[box] + " clips")
    mark(probes, probe)
    insideHead.textContent = "Inside box " + box
    explanation.textContent = "Loading the songs…"
    tiles.replaceChildren()
    detail.replaceChildren()
    picker.replaceChildren()
    picker.hidden = true
    try {
      const block = await loadBoxes(Math.floor(box / data.boxesPerFile))
      if (myVersion !== version) return
      shown = block[box % data.boxesPerFile]
      explanation.textContent =
        fmt(data.counts[box]) +
        " clips from " +
        fmt(shown.length) +
        " songs. Each tile holds one song. Each dot is a ten-second piece. Pick a tile to see its times."
      shown.forEach(([id, title, artist, , ...starts], i) => {
        const b = button("", () => selectSong(i), "fp-tile")
        const info =
          title + " · " + artist + " · " + starts.length + (starts.length === 1 ? " piece" : " pieces")
        b.title = info
        b.setAttribute("aria-label", info)
        b.dataset.track = id
        starts.forEach(() => b.append(el("i")))
        b.onkeydown = (e) => {
          const delta = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]
          if (delta === undefined && e.key !== "Home" && e.key !== "End") return
          e.preventDefault()
          const j =
            e.key === "Home"
              ? 0
              : e.key === "End"
                ? shown.length - 1
                : (i + delta + shown.length) % shown.length
          selectSong(j)
          tiles.children[j].focus()
        }
        tiles.append(b)
        picker.add(new Option(title + " · " + artist, String(i)))
      })
      picker.hidden = !shown.length
      if (shown.length) {
        const sourceGroup = shown.findIndex((g) => g[0] === example.source)
        selectSong(Math.max(sourceGroup, 0))
      }
    } catch {
      if (myVersion !== version) return
      explanation.textContent = "The songs in this box couldn’t load."
      detail.append(button("Try again", () => inspect(selected), "fp-quiet"))
    }
  }
  example.probes.forEach((b, i) =>
    probes.append(button((i ? "Extra box " : "Clip’s box ") + b, () => inspect(b))),
  )
  picker.onchange = () => selectSong(picker.selectedIndex)
  slider.oninput = () => inspect(Number(slider.value))
  canvas.onclick = (e) => {
    const r = canvas.getBoundingClientRect()
    inspect(Math.max(0, Math.min(1023, Math.floor(((e.clientX - r.left) / r.width) * 1024))))
  }
  const observer = new ResizeObserver(draw)
  observer.observe(canvas)
  onDispose(() => {
    version++
    observer.disconnect()
  })
  panel.append(probes, field, inside)
  inspect(initial)
  return panel
}

function timelinePanel(example, data) {
  const panel = el("div")
  panel.append(
    heading(
      "Find the matching moment.",
      "Across " +
        data.maps +
        " maps, " +
        fmt(example.candidates) +
        " of " +
        fmt(data.windows) +
        " saved pieces reached the final comparison. Each piece remembers where it came from.",
    ),
  )
  const switches = group("Compare song timelines", "fp-probes"),
    ids = [example.winner, example.runner]
  const area = el("figure", undefined, "fp-timeline"),
    detail = el("p", undefined, "fp-timeline-detail")
  const origins = el("div", undefined, "fp-origins")
  function view(idx) {
    mark(switches, idx)
    const id = ids[idx],
      track = example.timelines[id],
      source = id === example.source
    const title = el("h4", track.title),
      subtitle = el("p", track.artist + " · " + seconds(track.duration) + " seconds in the archive")
    const board = el("div", undefined, "fp-board")
    const axis = el("div", undefined, "fp-time-axis")
    for (let tick = 0; tick < track.duration - 2; tick += 5) {
      const t = el("span", tick + " s")
      t.style.left = (tick / track.duration) * 100 + "%"
      axis.append(t)
    }
    const end = el("span", Math.round(track.duration) + " s")
    end.className = "fp-end"
    axis.append(end)
    board.append(axis)
    const rows = group("Saved ten-second pieces", "fp-window-rows")
    const trackRows = []
    function select(i) {
      const w = track.windows[i]
      const ribbon = el("div", undefined, "fp-map-ribbon")
      ribbon.setAttribute("role", "img")
      ribbon.setAttribute("aria-label", "Maps that found this piece: " + (w.maps.join(", ") || "none"))
      for (let m = 1; m <= data.maps; m++) {
        const cell = el("i")
        cell.classList.toggle("hit", w.maps.includes(m))
        cell.title = "Map " + m + " · " + (w.maps.includes(m) ? "found this piece" : "missed this piece")
        ribbon.append(cell)
      }
      const answer = w.maps.length
        ? "It appeared on " +
          w.maps.length +
          " of " +
          data.maps +
          " maps. Map " +
          w.maps[0] +
          " found it first."
        : "Every map missed this piece, so the search skipped it."
      origins.replaceChildren(el("strong", "Which maps found this piece?"), ribbon, el("p", answer))
      trackRows.forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)))
      const text =
        w.score === null
          ? "This piece stayed outside the shortlist."
          : (w.best ? "Closest piece in the whole search. " : "") +
            "Match score " +
            w.score.toFixed(3) +
            " on a scale where 1 is a perfect match."
      detail.textContent =
        seconds(w.start) + "–" + seconds(w.start + data.windowSeconds) + " seconds · " + text
    }
    track.windows.forEach((w, i) => {
      const row = el("div", undefined, "fp-window-lane")
      const b = button(
        seconds(w.start) + "–" + seconds(w.start + data.windowSeconds) + " s",
        () => select(i),
        "fp-window",
      )
      b.setAttribute(
        "aria-label",
        b.textContent + (w.best ? ", closest piece" : w.score === null ? ", skipped" : ", compared"),
      )
      b.style.left = (100 * w.start) / track.duration + "%"
      b.style.width = (100 * data.windowSeconds) / track.duration + "%"
      b.classList.toggle("best", w.best)
      b.classList.toggle("skipped", w.score === null)
      row.append(b)
      rows.append(row)
      trackRows.push(b)
    })
    if (source) {
      const line = el("i", undefined, "fp-start-line")
      line.style.left = (example.start / track.duration) * 100 + "%"
      line.setAttribute("aria-hidden", "true")
      rows.append(line)
    }
    board.append(rows)
    const key = el(
      "figcaption",
      source
        ? "Dotted line: the test clip started here, at " + seconds(example.start) + " s."
        : "This is a different recording. Its clock starts at zero.",
    )
    key.append(" Tap a piece to see how it scored.")
    area.replaceChildren(title, subtitle, board, key)
    const best = track.windows.reduce(
      (best, w, i, a) => ((w.score ?? -Infinity) > (a[best].score ?? -Infinity) ? i : best),
      0,
    )
    select(best)
  }
  switches.append(
    button("Closest song", () => view(0)),
    button("Second-closest song", () => view(1)),
  )
  const verdict = el("div", undefined, "fp-verdict")
  verdict.append(
    el(
      "strong",
      example.accepted
        ? "The search named “" + example.timelines[example.winner].title + "”."
        : "The search kept its guess to itself.",
    ),
    el(
      "p",
      example.accepted
        ? "The closest piece sounded enough like the test clip, and the next song was far enough behind."
        : "“" +
            example.timelines[example.winner].title +
            "” came closest, but its best piece scored " +
            example.score.toFixed(3) +
            ". This test needed at least 0.750 to give a title.",
    ),
  )
  panel.append(switches, area, detail, origins, verdict)
  view(0)
  return panel
}

export async function mountRuler(container) {
  const response = await fetch("/ruler.json")
  if (!response.ok) throw Error("Missing song map")
  const data = await response.json()
  let example = data.cases[0],
    stage = 0,
    check = 0,
    box = example.probes[0],
    dispose
  container.className = "fingerprint-lab"
  const choices = group("Choose a test clip", "hr-choices"),
    tabs = group("Steps in the search", "fp-stages")
  const article = el("div", undefined, "fp-panel")
  article.id = "fp-panel"
  const footer = el("div", undefined, "fp-nav"),
    back = button("← Back", () => go(stage - 1, true), "fp-quiet"),
    forward = button("", () => go((stage + 1) % stages.length, true), "fp-primary")
  footer.append(back, forward)
  function go(i, focus = false) {
    stage = i
    dispose?.()
    dispose = undefined
    ;[...tabs.children].forEach((b, j) => {
      if (i === j) b.setAttribute("aria-current", "step")
      else b.removeAttribute("aria-current")
    })
    let content
    if (i === 0) content = soundPanel(example)
    if (i === 1)
      content = checksPanel(
        example,
        data,
        check,
        (v) => (check = v),
        (value) => {
          box = value
          go(2, true)
        },
      )
    if (i === 2)
      content = boxBrowser(
        example,
        data,
        box,
        (value) => (box = value),
        (fn) => (dispose = fn),
      )
    if (i === 3) content = timelinePanel(example, data)
    article.replaceChildren(content)
    back.hidden = i === 0
    forward.hidden = i === 1
    forward.textContent = i === 3 ? "Back to the sound ↺" : stages[i + 1] + " →"
    if (focus) {
      tabs.children[i].focus({ preventScroll: true })
      tabs.scrollIntoView({ block: "nearest" })
    }
  }
  data.cases.forEach((c, i) =>
    choices.append(
      button(c.timelines[c.source].title, () => {
        example = c
        check = 0
        box = c.probes[0]
        mark(choices, i)
        go(stage)
      }),
    ),
  )
  stages.forEach((name, i) => {
    const b = button("", () => go(i))
    b.setAttribute("aria-controls", article.id)
    b.append(el("span", String(i + 1)), el("b", name))
    tabs.append(b)
  })
  container.replaceChildren(el("h3", "Choose a clip", "fp-choose-title"), choices, tabs, article, footer)
  mark(choices, 0)
  go(0)
}
