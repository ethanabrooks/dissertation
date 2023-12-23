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
  show heading.where(level: 1): it => pagebreak(weak: true) + formatHeader(body: it)
  show figure: it => box[
    #it.body
    #pad(x: 1cm)[#it.caption]
  ]

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
  let space = 1.5em
  set par(leading: space)
  show heading: set block(above: 2 * space, below: space)
  show par: set block(above: space, below: space)
  show link: underline
  show figure: set block(breakable: true)

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

#let expand-to-2D-alignment(ali, default) = {
  assert(type(ali) == alignment)

  if ali.y == none {
    default.y
  } else {
    ali.y
  } + if ali.x == none {
    default.x
  } else {
    ali.x
  }
}

#let side-cap-fig(
  fig: none,
  fig-width: none,
  caption-width: none,
  caption-pos: top + left,
  caption: [],
  label: none,
  gutter: 1em,
  supplement-position: "left",
  placement: none,
  ..args,
) = {
  // move and scale the caption by *styling* it!
  let cap-width = {
    let t = type(caption-width)
    if t == ratio {
      // re-scale the caption-width when it is a ratio (i.e. in %) so that it applies to the width of the figure parent - not the image!
      as-ratio(caption-width / fig-width)
    } else if t == fraction {
      if type(fig-width) == fraction {
        // both widths in fr
        as-ratio(caption-width / fig-width)
      } else {
        // only one width in fr => make it full-width
        as-ratio((100% - fig-width) / fig-width) - gutter
      }
    } else if t == relative {
      // a relative length like "40% + 2em".
      // keep the 2em and re
      caption-width.length + eval(str(caption-width.ratio / fig-width * 100) + "%")
    } else {
      caption-width
    }
  }

  let ali = expand-to-2D-alignment(caption-pos, default-2D-align)
  // calculate the horizontal movement
  let dx = (cap-width + gutter) * {
    if ali.x == right {
      1
    } else {
      -1
    }
  }
  // apply the move+width+style
  show: caption-styles.with(supplement-position: supplement-position, container: (
    // (place, box),
    // ((ali,), ()),
    // ((dx: dx), (width: cap-width)))
    arguments(place, ali, dx: dx),
    arguments(box, width: cap-width),
  ))

  // actually this is going to be a grid again
  let columns = (
    [#figure(fig, caption: caption, ..args.named()) #label
    ],
    [], // the second column is not really necessary
  )
  // the original, user-defined caption width defines the column width!
  let col-widths = (fig-width, caption-width)
  // flip if the caption goes on the left
  if ali.x == left {
    col-widths = col-widths.rev()
    columns = columns.rev()
  }

  floaty(grid(columns: col-widths, gutter: gutter, ..columns), placement)
}

#let side-cap-fig(
  fig: none,
  fig-width: none,
  caption-width: none,
  caption-pos: top + left,
  caption: [],
  label: none,
  gutter: 1em,
  supplement-position: "left",
  placement: none,
  ..args,
) = {
  // move and scale the caption by *styling* it!
  let cap-width = {
    let t = type(caption-width)
    if t == ratio {
      // re-scale the caption-width when it is a ratio (i.e. in %) so that it applies to the width of the figure parent - not the image!
      as-ratio(caption-width / fig-width)
    } else if t == fraction {
      if type(fig-width) == fraction {
        // both widths in fr
        as-ratio(caption-width / fig-width)
      } else {
        // only one width in fr => make it full-width
        as-ratio((100% - fig-width) / fig-width) - gutter
      }
    } else if t == relative {
      // a relative length like "40% + 2em".
      // keep the 2em and re
      caption-width.length + eval(str(caption-width.ratio / fig-width * 100) + "%")
    } else {
      caption-width
    }
  }

  let ali = expand-to-2D-alignment(caption-pos, default-2D-align)
  // calculate the horizontal movement
  let dx = (cap-width + gutter) * {
    if ali.x == right {
      1
    } else {
      -1
    }
  }
  // apply the move+width+style
  show: caption-styles.with(supplement-position: supplement-position, container: (
    // (place, box),
    // ((ali,), ()),
    // ((dx: dx), (width: cap-width)))
    arguments(place, ali, dx: dx),
    arguments(box, width: cap-width),
  ))

  // actually this is going to be a grid again
  let columns = (
    [#figure(fig, caption: caption, ..args.named()) #label
    ],
    [], // the second column is not really necessary
  )
  // the original, user-defined caption width defines the column width!
  let col-widths = (fig-width, caption-width)
  // flip if the caption goes on the left
  if ali.x == left {
    col-widths = col-widths.rev()
    columns = columns.rev()
  }

  floaty(grid(columns: col-widths, gutter: gutter, ..columns), placement)
}