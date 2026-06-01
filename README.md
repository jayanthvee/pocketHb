# pocketHb

an open-source replication of the personalization layer from the [mannino et al PNAS 2025 paper](https://www.pnas.org/doi/abs/10.1073/pnas.2424677122) on smartphone hemoglobin estimation from fingernail photos.

the paper showed you can estimate someone's hemoglobin from a phone photo of their fingernails. the novel part wasn't the photo-to-hb mapping (plenty of people have done that). it was the *personalization*: give the model one real bloodwork reading as a baseline and it gets way more accurate for that specific person over time. sanguina licensed the tech and locked the code. i'm rebuilding the personalization piece in the open.

every existing OSS anemia repo i could find does single-image binary "anemic vs not anemic" classification. nobody's published the per-user calibration. that's the gap this fills.

base regressor is trained on the [nature sci data 2024 fingernail+hb dataset](https://www.nature.com/articles/s41597-024-03895-9). personalization is a small head fit per user against one known Hb value.

## status

in progress. day 1.

## not a medical device

research replication only. do not use this instead of an actual blood test. not FDA cleared, not validated clinically, not a doctor.
