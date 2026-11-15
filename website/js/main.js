import { mountReplay } from './replay.js';
import { mountWhatIf } from './whatif.js';
import { mountHeroViz } from './hero-viz.js';

mountHeroViz(document.getElementById('hero-viz'));
mountReplay(document.getElementById('console'));
mountWhatIf(document.getElementById('whatif'));
