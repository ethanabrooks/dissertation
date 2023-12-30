#import "math.typ": *
$ QValue_Highlight(k) (Obs_t, Act_t) := Rew(Obs_t, Act_t) + gamma E_(Obs_(t+1) Act_(t+1))
[ QValue_Highlight(k-1) (Obs_(t+1), Act_(t+1)) ] $