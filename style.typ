#let formatHeader(body: content) = [
  #set align(center)
  #body
  #v(18pt)
]
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
  set par(leading: 1.5em)
  show heading: set block(below: 1.5em)
  show par: set block(above: 1.5em)

  [ #author \
    #email \
    ORCID iD: #orcid \
    \
    © #author #year ]

  set align(left + top)
  set par(justify: true)
  set page(numbering: "i")
  show heading.where(level: 1): it => {
    pagebreak()
    formatHeader(body: it)
  }
  set par(first-line-indent: .5in)
  doc
}