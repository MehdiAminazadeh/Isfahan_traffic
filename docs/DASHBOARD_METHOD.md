# Dashboard method and figure map

The dashboard is for traffic planning and data review. It uses simple wording in
the visible descriptions. Scientific terms stay where they are needed and are
defined beside the figure.

## Metric rules

- A trusted intersection-month has at least 80% temporal coverage and at least
  80% mean operational-detector coverage.
- A paired change uses only intersections trusted in both compared months. The
  dashboard reports the median intersection change and the paired-site count.
- The daily 2019 index compares a day with the matching month and weekday median
  in 2019. A value of 100 matches that baseline.
- Functional graph edges mean similar 168-bin weekly traffic profiles. They are
  not road links.
- Approach values are entry-side detector groups, not exit or turning counts.

## Figure map

| Figure | Question | Form | Main fields | Main limit |
| --- | --- | --- | --- | --- |
| Network traffic by month | How did typical traffic move? | Line | month, median site volume | Site panel changes by month |
| Paired monthly change | Which monthly moves were largest? | Signed bar | paired median change, sites | Adjacent months only |
| 2019 month pattern | What was normal seasonality? | Bar | fixed-panel monthly median | One complete pre-pandemic year |
| Daily traffic index | When did daily traffic break from normal? | Line | 2019 index, 7-day median | Descriptive, not causal |
| Context differences | Which factors line up with traffic differences? | Horizontal bar | adjusted index points, days | Factors overlap |
| Shock and recovery | Did large falls predict large recoveries? | Scatter | Feb–Mar and Apr–May change | Paired trusted sites only |
| Largest March falls | Which sites fell most? | Horizontal bar | site change | Bottom twelve only |
| January year-over-year extremes | Which sites changed most by Jan 2021? | Horizontal bar | Jan 2021 vs Jan 2020 | Two-point comparison |
| High-volume site heatmap | Was the shock broad? | Heatmap | site-indexed monthly volume | Twelve selected sites |
| Weekly hour heatmap | When is normal traffic highest? | Heatmap | weekday, hour, mean volume | Archive-wide average |
| Intersection 1081 approaches | How did each entry direction move? | Multi-line | month, direction volume | No exits/turns |
| Busy-site direction mix | Which directions dominate? | 100% stacked bar | four approach shares | Four usable groups required |
| Volume versus centrality | Are busy sites also pattern-central? | Scatter | mean volume, weighted degree | Functional graph only |
| Community size | How large are the peer groups? | Bar | community, sites | Not geographic districts |
| Coordinate candidates | Where are mapped graph sites? | Spatial scatter | latitude, longitude, volume | No basemap; GIS review required |
| Source-month coverage | Which months are complete? | Bar | month, temporal coverage | Source completeness only |
| Trusted-month distribution | How much history does each site have? | Histogram | trusted months | Graph sites only |

## Saved validation

`outputs/tables/dashboard_validation.json` records key uniqueness, join, sample,
coverage, and missing-period checks. `outputs/tables/dashboard_insights.json`
stores the headline values used in the dashboard text.
