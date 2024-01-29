#let alg-counter = counter("algorithm")

#let show-algorithms(body) = {
  show figure.caption.where(kind: "algorithm"): it => it.body
  show figure.where(kind: "algorithm"): it => {
    alg-counter.step()
    counter(figure.where(kind: "line-number")).update(0)
    let title = it.caption
    figure(
      block(
        stroke: (y: 1pt),
        [
          #block(
            inset: (bottom: .2em, top: .3em),
            [*Algorithm #alg-counter.display()*] + { if title != none [*:* #title] },
          )
          #v(-1em) // Unfortunately, typst inserts a space here that we have to remove
          #block(
            inset: (y: .5em),
            width: 100%,
            stroke: (top: .7pt),
            align(left)[#it.body],
          )
        ],
      ),
      placement: it.placement,
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

#let render-comment(comment) = {
  if comment != none {
    set text(size: .7em, fill: luma(100))
    h(.001fr)
    sym.triangle.stroked.r + sym.space + comment
  }
}

#let render-line(line) = {
  assert(line.type == "line")
  pad(left: line.indent * 1em, line.words.map(render-word).join(" "))
}

#let render-line-number(line-number, line-label) = {
  set align(right)
  set text(size: .8em)
  [#box(
      [#figure([#line-number:], kind: "line-number", supplement: "line")
        #label(if line-label == none { "dummy-line-label" } else { str(line-label) })],
      height: .6em,
    )]
}

#let algorithm(..lines, line-numbers: true) = {
  let rows = lines.pos().enumerate(start: 1).map(((line-number, line)) => {
    (..if line-numbers {
      (render-line-number(line-number, line.label),)
    } else {
      ()
    }, render-line(line), render-comment(line.comment),)
  }).flatten()
  let columns = (..if line-numbers {
    (18pt,)
  } else {
    ()
  }, auto, 1fr)
  table(
    align: horizon,
    columns: columns,
    inset: (y: .3em),
    stroke: none,
    ..rows,
  )
}

#let kw = word => (body: word, keyword: true, type: "word")
#let nkw = word => (body: word, keyword: false, type: "word")
#let line = (..words, indent: 0, comment: none, label: none) => (
  words: words.pos(),
  indent: indent,
  type: "line",
  comment: comment,
  label: label,
)

#let check-line(line) = {
  let line-type = if type(line) == "dictionary" {
    line.at("type", default: type(line))
  } else {
    type(line)
  }
  assert(
    line-type == "line",
    message: "Expected line, but got " + line-type + {
      if line-type == "array" {
        "\nMake sure to add `..` before algorithm expressions that take multiple lines."
      } else { "" }
    },
  )
}
#let indent = (line) => {
  check-line(line)
  line + (indent: line.indent + 1)
}
#let indent-many = (..lines) => lines.pos().map(indent)
#let check-lines(..lines) ={
  for line in lines.pos() {
    check-line(line)
  }
}

#let State(body, ..args) = line(nkw(body), ..args)
#let Function(first-line, comment: none, label: none, ..body) = {
  check-lines(..body)
  (
    line(kw("function"), nkw(first-line), comment: comment, label: label),
    ..indent-many(..body),
    line(kw("end function")),
  )
}
#let Repeat(..body, cond, comment: none, label: none) = {
  check-lines(..body)
  (
    line(kw("repeat")),
    ..indent-many(..body),
    line(kw("until"), nkw(cond), comment: comment, label: label),
  )
}
#let While(cond, comment: none, label: none, ..body) = {
  check-lines(..body)
  (
    line(kw("while"), nkw(cond), kw("do"), comment: comment, label: label),
    ..indent-many(..body),
    line(kw("end while")),
  )
}
#let For(cond, comment: none, label: none, ..body) = {
  check-lines(..body)
  (
    line(kw("for"), nkw(cond), kw("do"), comment: comment, label: label),
    ..indent-many(..body.pos()),
    line(kw("end for")),
  )
}

#let If(predicate, ..consequent, comment: none, label: none) = {
  check-lines(..consequent)
  (
    line(kw("if"), nkw(predicate), kw("then"), comment: comment, label: label),
    ..indent-many(..consequent),
  )
}

#let Else(..consequent, comment: none, label: none) = {
  check-lines(..consequent)
  (
    line(kw("else"), comment: comment, label: label),
    ..indent-many(..consequent),
  )
}

#let Elif(predicate, ..consequent, comment: none, label: none) = {
  check-lines(..consequent)
  (
    line(kw("else if"), nkw(predicate), comment: comment, label: label),
    ..indent-many(..consequent),
  )
}

#let EndIf(..args) = line(kw("end if"), ..args)

#let Return(body, ..args) = line(kw("return"), nkw(body), ..args)
#let Input(body, ..args) = line(kw("input"), nkw(body), ..args)