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

#let render-comment(comment) = {
  if comment != none {
    set text(size: .7em, fill: gray)
    h(.001fr)
    sym.triangle.stroked.r + sym.space + comment
  }
}

#let render-line(line) = {
  assert(line.type == "line")
  pad(left: line.indent * 1em, line.words.map(render-word).join(" "))
}

#let render-line-number(line-number) = {
  set align(right + bottom)
  set text(size: .8em)
  [#line-number:]
}

#let algorithm(..lines, line-numbers: true) = {
  let rows = lines.pos().enumerate(start: 1).map(((line-number, line)) => {
    (..if line-numbers {
      (render-line-number(line-number),)
    } else {
      ()
    }, render-line(line), render-comment(line.comment),)
  }).flatten()
  let columns = (..if line-numbers {
    (18pt,)
  } else {
    ()
  }, auto, auto)
  table(columns: columns, inset: 0.3em, stroke: none, ..rows)
}

#let kw = word => (body: word, keyword: true, type: "word")
#let nkw = word => (body: word, keyword: false, type: "word")
#let line = (..words, indent: 0, comment: none) => (words: words.pos(), indent: indent, type: "line", comment: comment)

#let indent = (line) => {
  assert(line.type == "line")
  line + (indent: line.indent + 1)
}
#let indent-many = (..lines) => lines.pos().map(indent)

#let State(body, ..args) = line(nkw(body), ..args)
#let Function(first-line, comment: none, ..body) = {
  (
    line(kw("function"), nkw(first-line), comment: comment),
    ..indent-many(body),
    line(kw("end function")),
  )
}
#let Repeat(..body, cond, comment: none) = {
  (
    line(kw("repeat")),
    ..indent-many(body),
    line(kw("until"), nkw(cond), comment: comment),
  )
}
#let While(cond, comment: none, ..body) = {
  (
    line(kw("while"), nkw(cond), kw("do"), comment: comment),
    ..indent-many(body),
    line(kw("end while")),
  )
}
#let For(cond, comment: none, ..body) = {
  (
    line(kw("for"), nkw(cond), kw("do")),
    ..indent-many(..body.pos()),
    line(kw("end for")),
  )
}

#let If(predicate, ..consequent, comment: none) = (
  line(kw("if"), nkw(predicate), kw("then"), comment: comment),
  ..indent-many(..consequent),
)

#let Else(..consequent, comment: none) = (line(kw("else"), comment: comment), ..indent-many(..consequent))

#let Elif(predicate, ..consequent, comment: none) = (
  line(kw("else if"), nkw(predicate), comment: comment),
  ..indent-many(..consequent),
)

#let EndIf(..args) = line(kw("end if"), ..args)

#let Return(arg, ..args) = line(kw("return"), nkw(args), ..args)