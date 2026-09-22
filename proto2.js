(function() {
  const participantLog = document.getElementById('participant-log');
  const goalInput = document.getElementById('goal-input');
  const goalSend = document.getElementById('goal-send');

  const wizardQueue = document.getElementById('wizard-queue');
  const gapTimerEl = document.getElementById('gap-timer');
  const stepsToggle = document.getElementById('steps-toggle');
  const stepInput = document.getElementById('step-input');
  const stepDelay = document.getElementById('step-delay');
  const stepSend = document.getElementById('step-send');
  const goalInputLockNote = document.getElementById('lock-note');
  const stepsSentBox = document.getElementById('steps-sent');
  const resultField = document.getElementById('result-field');
  const resultSend = document.getElementById('result-send');
  const logTbody = document.getElementById('log-tbody');
  const copyLogBtn = document.getElementById('copy-log-btn');

  let pendingSince = null;
  let tickHandle = null;
  let goalCounter = 0;
  let stepCountThisGoal = 0;
  let currentGoalResultEl = null;
  let logRows = [];

  function fmtGap(ms) { return (ms / 1000).toFixed(1) + 's'; }

  function startTimer() {
    pendingSince = Date.now();
    if (tickHandle) clearInterval(tickHandle);
    tickHandle = setInterval(() => {
      gapTimerEl.textContent = 'gap: ' + fmtGap(Date.now() - pendingSince);
    }, 200);
  }

  function stopTimerAndLog(stepsShown, stepCount) {
    if (pendingSince === null) return;
    const gapMs = Date.now() - pendingSince;
    if (tickHandle) clearInterval(tickHandle);
    gapTimerEl.textContent = 'gap: —';
    pendingSince = null;
    goalCounter += 1;
    logRows.push({ n: goalCounter, gapSeconds: (gapMs / 1000).toFixed(1), stepsShown, stepCount });
    renderLog();
  }

  function renderLog() {
    logTbody.innerHTML = '';
    logRows.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + r.n + '</td><td>' + r.gapSeconds + '</td><td>' + (r.stepsShown ? 'on' : 'off') + '</td><td>' + r.stepCount + '</td>';
      logTbody.appendChild(tr);
    });
  }

  function addGoalLine(text) {
    const row = document.createElement('div');
    row.className = 'goal';
    row.textContent = 'Goal: ' + text;
    participantLog.appendChild(row);
    participantLog.scrollTop = participantLog.scrollHeight;
  }

  function addStepLine(text) {
    const row = document.createElement('div');
    row.className = 'step';
    row.textContent = '– ' + text;
    participantLog.appendChild(row);
    participantLog.scrollTop = participantLog.scrollHeight;
  }

  function addResultLine(text) {
    const row = document.createElement('div');
    row.className = 'result';
    row.textContent = 'Result: ' + text;
    participantLog.appendChild(row);
    participantLog.scrollTop = participantLog.scrollHeight;
  }

  function addQueueItem(text) {
    const empty = wizardQueue.querySelector('i');
    if (empty) empty.remove();
    const item = document.createElement('div');
    item.textContent = text;
    wizardQueue.appendChild(item);
    wizardQueue.scrollTop = wizardQueue.scrollHeight;
  }

  function addStepsSentLine(text) {
    const empty = stepsSentBox.querySelector('i');
    if (empty) empty.remove();
    const item = document.createElement('div');
    item.textContent = text;
    stepsSentBox.appendChild(item);
    stepsSentBox.scrollTop = stepsSentBox.scrollHeight;
  }

  function resetStepsSentBox() {
    stepsSentBox.innerHTML = '<i>steps sent this goal (visible to participant only if toggle on)</i>';
  }

  function submitGoal() {
    const text = goalInput.value.trim();
    if (!text) return;
    addGoalLine(text);
    addQueueItem(text);
    goalInput.value = '';
    goalInput.disabled = true;
    goalSend.disabled = true;
    goalInputLockNote.style.display = 'block';
    startTimer();
    stepCountThisGoal = 0;
    resetStepsSentBox();
    stepInput.disabled = false;
    stepDelay.disabled = false;
    stepSend.disabled = false;
    resultField.disabled = false;
    resultSend.disabled = false;
  }

  function sendStep() {
    const text = stepInput.value.trim();
    if (!text) return;
    const delaySec = parseFloat(stepDelay.value) || 0;
    stepCountThisGoal += 1;
    addStepsSentLine(text + (delaySec ? '  (+' + delaySec + 's)' : ''));
    if (stepsToggle.checked) {
      setTimeout(() => { addStepLine(text); }, delaySec * 1000);
    }
    stepInput.value = '';
    stepDelay.value = '';
  }

  function sendResult() {
    const text = resultField.value.trim();
    if (!text) return;
    addResultLine(text);
    stopTimerAndLog(stepsToggle.checked, stepCountThisGoal);
    resultField.value = '';
    stepInput.disabled = true;
    stepDelay.disabled = true;
    stepSend.disabled = true;
    resultField.disabled = true;
    resultSend.disabled = true;
    goalInput.disabled = false;
    goalSend.disabled = false;
    goalInputLockNote.style.display = 'none';
  }

  goalSend.addEventListener('click', submitGoal);
  goalInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') submitGoal(); });

  stepSend.addEventListener('click', sendStep);
  stepInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendStep(); });

  resultSend.addEventListener('click', sendResult);

  copyLogBtn.addEventListener('click', async () => {
    const header = 'n\tgap_seconds\tsteps_shown\tstep_count';
    const lines = logRows.map(r => r.n + '\t' + r.gapSeconds + '\t' + r.stepsShown + '\t' + r.stepCount);
    const out = [header].concat(lines).join('\n');
    try {
      await navigator.clipboard.writeText(out);
      copyLogBtn.textContent = 'Copied.';
      setTimeout(() => { copyLogBtn.textContent = 'Copy log as text'; }, 1200);
    } catch (err) {
      copyLogBtn.textContent = 'Copy failed — select manually';
    }
  });
})();