# Methodology

WildlifeBench evaluates complete labelling routes, not model names alone. A route includes the event set, image policy, allowed country context, prompt package, model/provider route, parser, scorer, result manifest and reporting policy.

The award-facing package has one redistributed result layer:

- SpeciesNet-overlap layer: 246 events / 82 species.

The SpeciesNet-overlap slice includes an event only when the ground-truth species is present in local SpeciesNet v4.0.3b classifier labels and is allowed by the SpeciesNet country geofence for that event. Excluded non-overlap events from the earlier 330-event development snapshot are not redistributed in this staging package.

Dataset-release metadata is provided through Croissant JSON-LD and a CamtrapDP companion package. These describe the released dataset snapshot, not individual API calls.
