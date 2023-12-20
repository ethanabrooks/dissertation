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
  set text(font: "Times New Roman")

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
  [ #author \
    #email \
    ORCID iD: #orcid \
    \
    © #author #year ]
  set align(left + top)
  set par(justify: true)

  show heading: it => [
    #pagebreak()
    #set align(center)
    #set text(12pt, weight: "bold")
    #v(72pt)
    #block(smallcaps(it.body))
    #v(18pt)
  ]

  doc
}