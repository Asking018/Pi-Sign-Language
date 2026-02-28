/**
 * Word Creator logic: rolling 10-frame 40% threshold, 3-second lock-in timer,
 * word display, backspace, clear, speak (Web Speech API).
 * Premium gate: wrap in isPremium check (handled in HTML/CSS + app.js).
 */
(function () {
    'use strict';

    const ROLLING_WINDOW = 10;       // last N frames
    const THRESHOLD_RATIO = 0.4;     // letter must appear in >= 40% of frames (4/10)
    const MIN_CONFIDENCE = 0.4;      // per-frame minimum confidence to count
    const LOCK_IN_MS = 3000;         // hold same letter for 3 seconds to lock in

    let frameBuffer = [];            // { letter, confidence }[], max length ROLLING_WINDOW
    let currentWord = '';
    let lockStartTime = null;        // when current dominant letter started (ms)
    let holdingLetter = null;        // letter we're holding for 3s (reset if dominant changes)
    let progressInterval = null;

    let wordDisplayEl, progressFillEl, progressTrackEl;
    let onProgressUpdate = null;     // optional callback(ratio 0..1)

    /**
     * Get dominant letter from rolling window: must appear in >= 40% of frames
     * and we use the most frequent letter among those with avg confidence >= MIN_CONFIDENCE.
     */
    function getDominantLetter() {
        if (frameBuffer.length < ROLLING_WINDOW) return null;
        const window = frameBuffer.slice(-ROLLING_WINDOW);
        const counts = {};
        const sumConf = {};
        for (const { letter, confidence } of window) {
            if (!letter || letter === '' || (typeof confidence === 'number' && confidence < MIN_CONFIDENCE)) continue;
            counts[letter] = (counts[letter] || 0) + 1;
            sumConf[letter] = (sumConf[letter] || 0) + confidence;
        }
        const minCount = Math.ceil(ROLLING_WINDOW * THRESHOLD_RATIO); // 4 of 10
        let bestLetter = null;
        let bestCount = 0;
        for (const [letter, count] of Object.entries(counts)) {
            if (count >= minCount && count > bestCount) {
                const avgConf = sumConf[letter] / count;
                if (avgConf >= MIN_CONFIDENCE) {
                    bestLetter = letter;
                    bestCount = count;
                }
            }
        }
        return bestLetter;
    }

    /**
     * Push one prediction and update lock-in state and progress bar.
     * @param {string} letter - Detected letter (e.g. 'A', ' ')
     * @param {number} confidence - 0..1
     */
    function onPrediction(letter, confidence) {
        if (letter != null && letter !== '') {
            frameBuffer.push({ letter: letter.trim(), confidence: typeof confidence === 'number' ? confidence : 0 });
            if (frameBuffer.length > ROLLING_WINDOW) frameBuffer.shift();
        } else {
            // No letter or hand lost: clear buffer and reset progress
            frameBuffer = [];
            resetLockProgress();
            return;
        }

        const dominant = getDominantLetter();
        const now = Date.now();

        if (dominant === null) {
            resetLockProgress();
            return;
        }

        // If dominant letter changed, restart 3-second timer
        if (holdingLetter !== null && dominant !== holdingLetter) {
            lockStartTime = now;
            holdingLetter = dominant;
        } else if (lockStartTime === null) {
            lockStartTime = now;
            holdingLetter = dominant;
        }

        const elapsed = now - lockStartTime;
        const ratio = Math.min(1, elapsed / LOCK_IN_MS);

        updateProgressBar(ratio);
        if (typeof onProgressUpdate === 'function') onProgressUpdate(ratio);

        if (ratio >= 1) {
            currentWord += holdingLetter;
            updateWordDisplay();
            resetLockProgress();
            holdingLetter = null;
        }
    }

    function resetLockProgress() {
        lockStartTime = null;
        holdingLetter = null;
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
        updateProgressBar(0);
        if (typeof onProgressUpdate === 'function') onProgressUpdate(0);
    }

    function updateProgressBar(ratio) {
        if (!progressFillEl || !progressTrackEl) return;
        const pct = Math.round(ratio * 100);
        progressFillEl.style.width = pct + '%';
        progressTrackEl.setAttribute('aria-valuenow', pct);
    }

    function updateWordDisplay() {
        if (!wordDisplayEl) return;
        wordDisplayEl.textContent = currentWord === '' ? '—' : currentWord;
    }

    function backspace() {
        if (currentWord.length > 0) {
            currentWord = currentWord.slice(0, -1);
            updateWordDisplay();
        }
    }

    function addSpace() {
        currentWord += ' ';
        updateWordDisplay();
    }

    function clearAll() {
        currentWord = '';
        frameBuffer = [];
        resetLockProgress();
        updateWordDisplay();
    }

    function speakWord() {
        const text = currentWord.trim();
        if (!text) return;
        try {
            const u = new SpeechSynthesisUtterance(text);
            u.lang = 'en-US';
            u.rate = 0.9;
            speechSynthesis.speak(u);
        } catch (e) {
            console.warn('SpeechSynthesis failed:', e);
        }
    }

    /**
     * Initialize DOM refs and button listeners. Call once after DOM ready.
     * @param {object} opts - { isPremium: boolean }
     */
    function init(opts) {
        opts = opts || {};
        wordDisplayEl = document.getElementById('word-display');
        progressFillEl = document.getElementById('capture-progress-fill');
        progressTrackEl = document.querySelector('.capture-progress-track');

        const spaceBtn = document.getElementById('word-space');
        const backspaceBtn = document.getElementById('word-backspace');
        const clearBtn = document.getElementById('word-clear');
        const speakBtn = document.getElementById('word-speak');

        if (spaceBtn) spaceBtn.addEventListener('click', addSpace);
        if (backspaceBtn) backspaceBtn.addEventListener('click', backspace);
        if (clearBtn) clearBtn.addEventListener('click', clearAll);
        if (speakBtn) speakBtn.addEventListener('click', speakWord);

        updateWordDisplay();
        updateProgressBar(0);
    }

    /**
     * Reset word creator state when switching mode (e.g. clear buffer so next time is fresh).
     */
    function reset() {
        frameBuffer = [];
        resetLockProgress();
        holdingLetter = null;
        updateProgressBar(0);
    }

    // Public API
    window.wordLogic = {
        onPrediction: onPrediction,
        init: init,
        reset: reset,
        addSpace: addSpace,
        backspace: backspace,
        clearAll: clearAll,
        speakWord: speakWord,
        getWord: function () { return currentWord; },
        setOnProgressUpdate: function (fn) { onProgressUpdate = fn; }
    };
})();
