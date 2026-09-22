(function() {
  const participantLog = document.getElementById('participant-log');
  const participantInput = document.getElementById('participant-input');
  const participantSend = document.getElementById('participant-send');

  const wizardQueue = document.getElementById('wizard-queue');
  const gapTimerEl = document.getElementById('gap-timer');
  const reasoningToggle = document.getElementById('reasoning-toggle');
  const reasoningField = document.getElementById('reasoning-field');
  const responseField = document.getElementById('response-field');
  const wizardSend = document.getElementById('wizard-send');
  const logTbody = document.getElementById('log-tbody');
  const copyLogBtn = document.getElementById('copy-log-btn');

  let pendingSince = null;
  let tickHandle = null;
  let msgCounter = 0;
  let logRows = [];
  let reasoningSentUpTo = 0;
  let currentReasoningContainer = null;

  reasoningToggle.addEventListener('change', () => {
    reasoningField.style.display = reasoningToggle.checked ? 'block' : 'none';
  });

  reasoningField.addEventListener('input', flushReasoningLines);

  function fmtGap(ms) { return (ms / 1000).toFixed(1) + 's'; }

  function startTimer() {
    pendingSince = Date.now();
    if (tickHandle) clearInterval(tickHandle);
    tickHandle = setInterval(() => {
      gapTimerEl.textContent = 'gap: ' + fmtGap(Date.now() - pendingSince);
    }, 200);
  }

  function stopTimerAndLog(reasoningOn) {
    if (pendingSince === null) return;
    const gapMs = Date.now() - pendingSince;
    if (tickHandle) clearInterval(tickHandle);
    gapTimerEl.textContent = 'gap: —';
    pendingSince = null;
    msgCounter += 1;
    logRows.push({ n: msgCounter, gapSeconds: (gapMs / 1000).toFixed(1), reasoningOn });
    renderLog();
  }

  function renderLog() {
    logTbody.innerHTML = '';
    logRows.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td>' + r.n + '</td><td>' + r.gapSeconds + '</td><td>' + (r.reasoningOn ? 'on' : 'off') + '</td>';
      logTbody.appendChild(tr);
    });
  }

  function addParticipantBubble(text) {
    const row = document.createElement('div');
    row.className = 'msg user';
    row.textContent = text;
    participantLog.appendChild(row);
    participantLog.scrollTop = participantLog.scrollHeight;
  }

  function addAiBubble(text) {
    const row = document.createElement('div');
    row.className = 'msg ai';
    row.textContent = text;
    participantLog.appendChild(row);
    participantLog.scrollTop = participantLog.scrollHeight;
  }

  function addReasoningStep(step) {
    if (!currentReasoningContainer) {
      currentReasoningContainer = document.createElement('div');
      currentReasoningContainer.className = 'msg ai reasoning';
      participantLog.appendChild(currentReasoningContainer);
    }
    const stepEl = document.createElement('div');
    stepEl.className = 'reasoning-step';
    stepEl.textContent = step;
    currentReasoningContainer.appendChild(stepEl);
    participantLog.scrollTop = participantLog.scrollHeight;
  }

  function flushReasoningLines() {
    if (!reasoningToggle.checked) return;
    const unsent = reasoningField.value.slice(reasoningSentUpTo);
    const newlineIdx = unsent.lastIndexOf('\n');
    if (newlineIdx === -1) return;
    reasoningField.value.slice(reasoningSentUpTo, reasoningSentUpTo + newlineIdx)
      .split('\n').map(s => s.trim()).filter(Boolean)
      .forEach(addReasoningStep);
    reasoningSentUpTo += newlineIdx + 1;
  }

  function addQueueItem(text) {
    const empty = wizardQueue.querySelector('i');
    if (empty) empty.remove();
    const item = document.createElement('div');
    item.textContent = text;
    wizardQueue.appendChild(item);
    wizardQueue.scrollTop = wizardQueue.scrollHeight;
  }

  function sendParticipantMessage() {
    const text = participantInput.value.trim();
    if (!text) return;
    addParticipantBubble(text);
    addQueueItem(text);
    participantInput.value = '';
    reasoningField.value = '';
    reasoningSentUpTo = 0;
    currentReasoningContainer = null;
    startTimer();
    wizardSend.disabled = false;
  }

  function sendWizardResponse() {
    const text = responseField.value.trim();
    if (!text) return;
    const reasoningOn = reasoningToggle.checked;
    if (reasoningOn) {
      const leftover = reasoningField.value.slice(reasoningSentUpTo).trim();
      if (leftover) addReasoningStep(leftover);
    }
    addAiBubble(text);
    stopTimerAndLog(reasoningOn);
    responseField.value = '';
    reasoningField.value = '';
    reasoningSentUpTo = 0;
    currentReasoningContainer = null;
    wizardSend.disabled = true;
  }

  participantSend.addEventListener('click', sendParticipantMessage);
  participantInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendParticipantMessage(); });

  wizardSend.addEventListener('click', sendWizardResponse);
  responseField.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) sendWizardResponse(); });

  copyLogBtn.addEventListener('click', async () => {
    const header = 'n\tgap_seconds\treasoning_on';
    const lines = logRows.map(r => r.n + '\t' + r.gapSeconds + '\t' + r.reasoningOn);
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