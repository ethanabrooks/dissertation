#let alg-counter = counter("algorithm")

#let show-algorithms(body) = {
  show figure.caption.where(kind: "algorithm"): it => it.body
  show figure.where(kind: "algorithm"): it => {
    alg-counter.step()
    let title = it.caption
    block(
      stroke: (y: 1pt),
      [
        #block(
          inset: (bottom: .2em, top: .3em),
          [*Algorithm #alg-counter.display()*] + { if title != none [*:* #title] },
        )
        #v(-1em) // Unfortunately, typst inserts a space here that we have to remove
        #block(inset: (y: .5em), stroke: (top: .7pt), align(left)[#it.body])
      ],
    )
  }
  body
}

#let algorithm-figure = figure.with(supplement: "Algorithm", kind: "algorithm")

#let render-keyword(keyword) = [*#keyword*]

#let render-word(expr) = {
  assert(expr.type == "word")
  if expr.keyword {
    render-keyword(expr.body)
  } else {
    expr.body
  }
}

#let render-line(expr) = {
  assert(expr.type == "line")
  pad(left: expr.indent * 1em)[
    #expr.words.map(render-word).join(" ")
  ]
}

#let render-line-number(line-number) = {
  set align(right + bottom)
  set text(size: .8em)
  [#line-number:]
}

#let render-with-line-numbers = (..lines) => {
  let rows = lines.pos().enumerate(start: 1).map(((line-number, line)) => {
    (render-line-number(line-number), render-line(line))
  }).flatten()
  table(columns: (18pt, 100%), inset: 0.3em, stroke: none, ..rows)
}

#let render-without-line-numbers = (..lines) => {
  let rows = lines.pos().map(render-line)
  table(columns: (100%), inset: 0.3em, stroke: none, ..rows)
}

#let algorithm(..lines, line-numbers: true) = {
  if line-numbers {
    render-with-line-numbers(..lines)
  } else {
    render-without-line-numbers(..lines)
  }
}

#let kw = word => (body: word, keyword: true, type: "word")
#let nkw = word => (body: word, keyword: false, type: "word")
#let line = (..words, indent: 0) => (words: words.pos(), indent: indent, type: "line")

#let indent = (line) => line + (indent: line.indent + 1)
#let indent-many = (..lines) => lines.pos().map(indent)

#let State(body) = line(nkw(body), indent: 0)
#let Function(first-line, ..body) = {
  (
    line(kw("function"), nkw(first-line)),
    ..indent-many(body),
    line(kw("end function")),
  )
}
#let Repeat(..body, cond) = {
  (line(kw("repeat")), ..indent-many(body), line(kw("until"), nkw(cond)),)
}
#let While(cond, ..body) = {
  (
    line(kw("while"), nkw(cond), kw("do")),
    ..indent-many(body),
    line(kw("end while")),
  )
}
#let For(cond, ..body) = {
  (
    line(kw("for"), nkw(cond), kw("do")),
    ..indent-many(..body.pos()),
    line(kw("end for")),
  )
}

#let If(cond, body, ..elseif, els: none) = {
  let elseif-lines = elseif.pos().map(e =>
  (line(kw("else if"), nkw(e), kw("then")), ..indent-many(e)))
  let else-line = if els == none {
    ()
  } else {
    (line(kw("else")), ..indent-many(els))
  }
  (
    line(kw("if"), nkw(cond), kw("then")),
    ..indent-many(body),
    ..elseif-lines,
    ..else-line,
    line(kw("end if")),
  )
}
#let Return(arg) = line(kw("return"), nkw(args))