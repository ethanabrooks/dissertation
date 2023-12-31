#import "@preview/anti-matter:0.1.1": anti-matter, fence, set-numbering

#let formatHeader(body: content) = [
  #set align(center)
  #body
  #v(18pt)
]

#let cites(..labels) = {
  labels.pos().map(cite).join()
}

#let style(
  title: str,
  author: str,
  department: str,
  date: datetime,
  email: content,
  orcid: str,
  committee: (),
  doc: content,
) = {
  set document(title: title, author: author, date: date)
  set page(margin: (x: 1in))
  set align(center)
  set text(font: "New Computer Modern")
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => pagebreak(weak: true) + formatHeader(body: it)

  show link: underline

  v(108pt)
  [*#title*]
  let year = date.year()
  [ \
    \
    by \
    \
    #author \
    #v(72pt)\
    A dissertation submitted in partial fulfillment \
    of the requirements for the degree of \
    Doctor of Philosophy \
    (#department)\
    in the University of Michigan\
    #year\ ]
  v(108pt)
  set align(left)
  par(hanging-indent: 1in)[
    Doctoral Committee: \
    #committee.map(member => {
      linebreak()
      member
    }).join()
  ]
  pagebreak()
  set align(center + horizon)
  let space = 2em
  set par(leading: space, first-line-indent: 0.5in)
  show heading: set block(above: 2 * space, below: space)
  show par: set block(above: space, below: space)
  show link: underline
  show figure: set block(breakable: true)
  show figure: set par(leading: 1em)
  set math.equation(numbering: "(1)")

  [ #author \
    #email \
    ORCID iD: #orcid \
    \
    © #author #year ]

  set align(left + top)
  set par(justify: true)
  set page(numbering: "i")

  doc
}
