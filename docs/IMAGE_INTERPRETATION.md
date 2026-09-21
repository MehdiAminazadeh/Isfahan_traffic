# Reading the SCATS site diagrams

The diagrams are control-site schematics, not ordinary map tiles.

- `TCS ####` is the SCATS intersection/site identifier.
- Colored numbered rectangles are detector channel IDs.
- The rectangles sit on inbound lanes close to the stop line.
- Grey lane arrows show permitted movement direction.
- Magenta numbers and letters relate to signal control/phasing and must not be parsed as vehicle counts.
- Cyan four-digit labels at road ends may identify adjacent SCATS sites, but those links require verification before they are used as physical network edges.

For 1081, the detector groups are south `1–3`, west `4–6`, north `7–9`, and east `10–12`. A raw record such as `Int 1081 1=14 ... 12=0` therefore contains 12 approach-channel counts for one 15-minute timestamp.

The values measure entries toward the stop line. An exit from 1081 can only be estimated after connecting a movement to a downstream detector or neighboring SCATS site. The current project uses the phrase **approach entry volume** and keeps the functional similarity graph distinct from a physical road graph.
