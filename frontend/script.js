const API_BASE = '/api';

const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const searchBtn = document.getElementById('searchBtn');
const textSearchInput = document.getElementById('textSearchInput');
const micBtn = document.getElementById('micBtn');
const textSearchBtn = document.getElementById('textSearchBtn');
const uploadedPreview = document.getElementById('uploadedPreview');

// Auto-resize textarea like ChatGPT
textSearchInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});
const exactSection = document.getElementById('exactSection');
const similarSection = document.getElementById('similarSection');
const exactResultsList = document.getElementById('exactResultsList');
const similarResultsList = document.getElementById('similarResultsList');
const statusText = document.getElementById('statusText');
const resultTemplate = document.getElementById('resultTemplate');
const infoBar = document.getElementById('infoBar');
const infoCategory = document.getElementById('infoCategory');
const infoBestMatch = document.getElementById('infoBestMatch');
const infoResults = document.getElementById('infoResults');
const infoTime = document.getElementById('infoTime');
const loadMoreWrapper = document.getElementById('loadMoreWrapper');
const loadMoreBtn = document.getElementById('loadMoreBtn');
const otherModelsTrigger = document.getElementById('otherModelsTrigger');
const showOtherModelsBtn = document.getElementById('showOtherModelsBtn');
const otherModelsSection = document.getElementById('otherModelsSection');
const otherResultsList = document.getElementById('otherResultsList');
const loadMoreOtherWrapper = document.getElementById('loadMoreOtherWrapper');
const loadMoreOtherBtn = document.getElementById('loadMoreOtherBtn');

let selectedFile = null;
let currentSimilarMatches = [];
let currentSimilarIndex = 0;
let currentOtherMatches = [];
let currentOtherIndex = 0;
const ITEMS_PER_PAGE = 5;

let isSearching = false;

// Tracks if search is 'image' or 'text'/'voice'
let currentSearchType = 'image';
let compareModeActive = false;
let compareImage1 = null;
let compareImage2 = null;
// Dedicated variable for the uploaded image's server-resolvable URL.
// Set when image search completes so startAutomaticComparison() always
// has the right Image 1 regardless of whether an exact DB match was found.
let uploadedImageServerUrl = null;

function setStatus(message) {
  statusText.textContent = message;
}

function clearUploadedPreview() {
  uploadedPreview.classList.add('empty');
  uploadedPreview.innerHTML = '<span class="preview-placeholder">Image Preview</span>';
  const sb = document.getElementById('unifiedSearchBar');
  if (sb) sb.classList.remove('has-preview');
}

function showUploadedPreview(file) {
  const imageUrl = URL.createObjectURL(file);
  uploadedPreview.classList.remove('empty');
  uploadedPreview.innerHTML = '';

  const image = document.createElement('img');
  image.src = imageUrl;
  image.alt = 'Uploaded preview';

  uploadedPreview.appendChild(image);
  
  const fnElem = document.getElementById('previewFilename');
  if (fnElem) fnElem.textContent = file.name;
  
  const sb = document.getElementById('unifiedSearchBar');
  if (sb) sb.classList.add('has-preview');
}

function clearResults() {
  exactSection.style.display = 'none';
  similarSection.style.display = 'none';
  if (otherModelsSection) otherModelsSection.style.display = 'none';
  if (otherModelsTrigger) otherModelsTrigger.style.display = 'none';
  exactResultsList.innerHTML = '';
  similarResultsList.innerHTML = '';
  if (otherResultsList) otherResultsList.innerHTML = '';
  infoBar.style.display = 'none';
}

function createResultCard(result) {
  const card = resultTemplate.content.firstElementChild.cloneNode(true);
  const image = card.querySelector('.result-image');
  const score = card.querySelector('.result-score');

  // Encode path segments to handle spaces, '#' and other special characters
  let encodedPath = '';
  if (result.image_url) {
      encodedPath = result.image_url.split('/').map(encodeURIComponent).join('/');
      // Keep drive letter colon intact if present
      encodedPath = encodedPath.replace(/%3A/g, ':');
  }
  const imageUrl = result.image_data_url || `${API_BASE}/images/${encodedPath}`;
  image.src = imageUrl;
  
  const heartBtn = card.querySelector('.heart');
  if (heartBtn) {
      let myCollections = JSON.parse(localStorage.getItem('myCollections')) || [];
      if (myCollections.includes(imageUrl)) {
          heartBtn.textContent = '♥';
      }
      
      heartBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          let myColls = JSON.parse(localStorage.getItem('myCollections')) || [];
          const idx = myColls.indexOf(imageUrl);
          if (idx > -1) {
              myColls.splice(idx, 1);
              heartBtn.textContent = '♡';
          } else {
              myColls.push(imageUrl);
              heartBtn.textContent = '♥';
          }
          localStorage.setItem('myCollections', JSON.stringify(myColls));
          if (window.updateCollectionCount) window.updateCollectionCount();
      });
  }
  
  // Format score.
  // match_percent is calibrated server-side from the measured score
  // distribution (see calibrate_score in services/similarity_search.py).
  // The old code rounded raw cosine and clamped anything above 0.99 to 100%,
  // which labelled every result a 100% match.
  const percentage = Math.round(
      result.match_percent != null ? result.match_percent : result.similarity * 100
  );
  score.textContent = `${percentage}% Match`;

  if (currentSearchType !== 'image') {
      score.style.display = 'none';
  }

  const countBadge = card.querySelector('.result-count');
  if (countBadge) {
      const quantity = result.count || 1;
      if (quantity > 1) {
          countBadge.textContent = `Quantity - ${quantity}`;
          countBadge.style.display = 'flex';
      } else {
          countBadge.style.display = 'none';
      }
  }

  // Populate metadata (category + features)
  const metaSection = card.querySelector('.result-meta');
  const metaCategory = card.querySelector('.meta-category');
  const metaFeatures = card.querySelector('.meta-features');
  
  if (metaSection && (result.meta_category || (result.meta_features && result.meta_features.length > 0))) {
      metaSection.style.display = 'block';
      if (result.meta_category && metaCategory) {
          metaCategory.textContent = result.meta_category;
      }
      if (result.meta_features && result.meta_features.length > 0 && metaFeatures) {
          metaFeatures.innerHTML = result.meta_features
              .map(f => `<span class="meta-tag">${f}</span>`)
              .join('');
      }
  }

  const compareTriggerBtn = card.querySelector('.compare-trigger-btn');
  if (compareTriggerBtn) {
      compareTriggerBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          enterCompareMode();
      });
  }

  card.addEventListener('click', async (e) => {
    if (e.target.closest('.heart')) return;
    if (e.target.closest('.compare-trigger-btn')) return;

    if (compareModeActive) {
        document.querySelectorAll('.selected-for-compare').forEach(el => el.classList.remove('selected-for-compare'));
        card.classList.add('selected-for-compare');
        compareImage2 = imageUrl;
        if (typeof window.startAutomaticComparison === 'function') window.startAutomaticComparison();
        return;
    }

    // Click to search again using this image
    if (window.searchFromImageUrl) {
        window.searchFromImageUrl(imageUrl);
    }
  });

  return card;
}



function _dedupeByScore(list) {
  // Group DB entries that share the exact same similarity score (the same
  // underlying image indexed under more than one filename/path) so they
  // render once with a "Quantity - N" badge instead of as visual duplicates.
  const grouped = [];
  const seen = new Map();
  list.forEach(result => {
    const simKey = result.similarity.toFixed(5);
    if (!seen.has(simKey)) {
      result.count = 1;
      seen.set(simKey, result);
      grouped.push(result);
    } else {
      seen.get(simKey).count++;
    }
  });
  return grouped;
}

function renderExactMatchCard(file, category, uploadedFilename, features) {
  // The true "exact match" for an image search is the file the user just
  // uploaded, always 100% - never a database result. Rendered from the
  // in-memory file directly (via an object URL) rather than anything the
  // backend returns.
  exactResultsList.innerHTML = '';
  if (!file) {
    exactSection.style.display = 'none';
    return;
  }

  const card = resultTemplate.content.firstElementChild.cloneNode(true);
  const image = card.querySelector('.result-image');
  const score = card.querySelector('.result-score');
  const heartBtn = card.querySelector('.heart');
  const compareBtn = card.querySelector('.compare-trigger-btn');
  const countBadge = card.querySelector('.result-count');
  const metaSection = card.querySelector('.result-meta');
  const metaCategory = card.querySelector('.meta-category');
  const metaFeatures = card.querySelector('.meta-features');

  image.src = URL.createObjectURL(file);
  if (uploadedFilename) {
      image.dataset.serverUrl = `/api/images/backend/uploads/${uploadedFilename}`;
  }
  image.alt = 'Uploaded image (exact match)';
  score.textContent = '100% Match';
  if (countBadge) countBadge.style.display = 'none';

  // Heart (collections) and Compare both key off a stable, persistent
  // image_url - the uploaded file only has an ephemeral blob: URL for this
  // browser session, so those actions don't apply here.
  if (heartBtn) heartBtn.style.display = 'none';
  if (compareBtn) compareBtn.style.display = 'none';

  if (metaSection && (category || (features && features.length > 0))) {
    metaSection.style.display = 'block';
    if (category && metaCategory) metaCategory.textContent = category;
    if (features && features.length > 0 && metaFeatures) {
      metaFeatures.innerHTML = features
          .map(f => `<span class="meta-tag">${f}</span>`)
          .join('');
    }
  }

  exactResultsList.appendChild(card);
  exactSection.style.display = 'block';
}

function renderImageSearchResults(similarResults, otherResults, searchTimeMs, category) {
  // Image search only: the backend already splits results into
  // similar_results/other_model_results (band=="similar" / band=="other"),
  // with any near-duplicate of the uploaded image itself already excluded
  // server-side. No client-side re-grouping needed - just render the two
  // sections in the required fixed order, below the exact-match card.
  similarResultsList.innerHTML = '';
  if (otherResultsList) otherResultsList.innerHTML = '';
  similarSection.style.display = 'none';
  if (otherModelsSection) otherModelsSection.style.display = 'none';
  infoBar.style.display = 'flex';

  const timeSec = (searchTimeMs / 1000).toFixed(2);

  const similarMatches = _dedupeByScore(similarResults || []);
  const otherMatches = _dedupeByScore(otherResults || []);

  const similarSectionTitle = similarSection.querySelector('.section-divider');
  if (similarSectionTitle) similarSectionTitle.textContent = '✨ Similar Results ✨';

  currentSimilarMatches = similarMatches;
  currentSimilarIndex = 0;
  currentOtherMatches = otherMatches;
  currentOtherIndex = 0;

  if (loadMoreWrapper) loadMoreWrapper.style.display = 'none';
  if (otherModelsTrigger) otherModelsTrigger.style.display = 'none';
  if (loadMoreOtherWrapper) loadMoreOtherWrapper.style.display = 'none';

  if (similarMatches.length > 0) {
    similarSection.style.display = 'block';
    renderNextBatch();
  } else if (otherMatches.length > 0) {
    // No results in the "similar" band, but weaker matches exist - let the
    // user reveal them via the existing Other Models trigger rather than
    // showing an empty Similar Ornaments section.
    similarSection.style.display = 'block';
    if (otherModelsTrigger) otherModelsTrigger.style.display = 'block';
  }

  infoCategory.textContent = category || 'Unknown';

  const bestMatchItem = infoBestMatch.closest('.info-item');
  const foundItem = infoResults.closest('.info-item');

  const bestPool = similarMatches.length > 0 ? similarMatches : otherMatches;
  const best = bestPool[0];
  infoBestMatch.textContent = best
    ? `${Math.round(best.match_percent != null ? best.match_percent : best.similarity * 100)}%`
    : '0%';
  if (bestMatchItem) bestMatchItem.style.display = 'flex';
  if (foundItem) foundItem.style.display = 'flex';

  infoResults.textContent = `${similarMatches.length} Images`;
  infoTime.textContent = `${timeSec} sec`;

  const istTime = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const infoTimestamp = document.getElementById('infoTimestamp');
  if (infoTimestamp) infoTimestamp.textContent = `${istTime} IST`;
}

function renderResults(results, searchTimeMs, category) {
  exactResultsList.innerHTML = '';
  similarResultsList.innerHTML = '';
  exactSection.style.display = 'none';
  similarSection.style.display = 'none';
  infoBar.style.display = 'flex';

  const timeSec = (searchTimeMs / 1000).toFixed(2);

  if (!results.length) {
    infoBestMatch.textContent = '0%';
    infoResults.textContent = '0 Images';
    infoTime.textContent = `${timeSec} sec`;
    infoCategory.textContent = category || 'Unknown';
    return;
  }

  // Deduplicate by similarity score (to 5 decimal places) to group identical images that might have different filenames
  const groupedResults = [];
  const seenSims = new Map();

  results.forEach(result => {
    // Identical images will have the exact same vector and thus the exact same similarity score
    const simKey = result.similarity.toFixed(5);
    
    if (!seenSims.has(simKey)) {
      result.count = 1;
      seenSims.set(simKey, result);
      groupedResults.push(result);
    } else {
      seenSims.get(simKey).count++;
    }
  });

  let finalResults = [];
  let otherMatches = [];
  let exactMatches = [];
  let similarMatches = [];

  if (currentSearchType === 'image') {
      // The server assigns each hit a band ("exact" / "similar" / "other")
      // from calibrated thresholds, so the cut-offs live in one place instead
      // of being duplicated here as raw-cosine magic numbers.
      // Combined image+text search merges scores client-side and can hand us
      // rows with no band, so fall back to the percentage (and finally to
      // "similar") rather than dropping them from every group.
      const bandOf = (r) => {
          if (r.band) return r.band;
          if (r.match_percent == null) return 'similar';
          if (r.match_percent >= 97) return 'exact';
          return r.match_percent >= 85 ? 'similar' : 'other';
      };
      exactMatches = groupedResults.filter(r => bandOf(r) === 'exact');
      similarMatches = groupedResults.filter(r => bandOf(r) === 'similar');
      otherMatches = groupedResults.filter(r => bandOf(r) === 'other');
      finalResults = exactMatches.concat(similarMatches);
  } else {
      // For Text / Voice Search: No percentage filtering, no Other Models
      finalResults = groupedResults;
      otherMatches = [];
      exactMatches = [];
      similarMatches = groupedResults;
      
      // Optionally rename "Similar Results" section title for text search
      const similarSectionTitle = similarSection.querySelector('.section-divider');
      if (similarSectionTitle) {
          similarSectionTitle.textContent = '✨ Search Results ✨';
      }
  }

  if (currentSearchType === 'image') {
      const similarSectionTitle = similarSection.querySelector('.section-divider');
      if (similarSectionTitle) {
          similarSectionTitle.textContent = '✨ Similar Results ✨';
      }
  }

  if (exactMatches.length > 0) {
    exactSection.style.display = 'block';
    exactMatches.forEach(result => {
        exactResultsList.appendChild(createResultCard(result));
    });
  }

  currentSimilarMatches = similarMatches;
  currentSimilarIndex = 0;
  
  currentOtherMatches = otherMatches;
  currentOtherIndex = 0;
  
  if (loadMoreWrapper) loadMoreWrapper.style.display = 'none';
  if (otherModelsTrigger) otherModelsTrigger.style.display = 'none';
  if (loadMoreOtherWrapper) loadMoreOtherWrapper.style.display = 'none';

  if (similarMatches.length > 0) {
    similarSection.style.display = 'block';
    renderNextBatch();
  } else if (otherMatches.length > 0) {
    // If no similar matches but there are other models
    similarSection.style.display = 'block';
    if (otherModelsTrigger) otherModelsTrigger.style.display = 'block';
  }

  // Populate Info Bar
  infoCategory.textContent = category || 'Unknown';
  
  const bestMatchItem = infoBestMatch.closest('.info-item');
  const foundItem = infoResults.closest('.info-item');

  if (currentSearchType === 'image' && results.length > 0) {
      const best = results[0];
      infoBestMatch.textContent = `${Math.round(
          best.match_percent != null ? best.match_percent : best.similarity * 100
      )}%`;
      if (bestMatchItem) {
          bestMatchItem.style.display = 'flex';
      }
  } else {
      if (bestMatchItem) {
          bestMatchItem.style.display = 'none';
      }
  }

  if (foundItem) foundItem.style.display = 'flex';
  
  // Show total matching images found
  infoResults.textContent = `${finalResults.length} Images`;
  infoTime.textContent = `${timeSec} sec`;
  
  const istTime = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute:'2-digit', second:'2-digit' });
  const infoTimestamp = document.getElementById('infoTimestamp');
  if (infoTimestamp) {
      infoTimestamp.textContent = `${istTime} IST`;
  }
}

async function runSearch() {
  if (isSearching) return;
  
  if (!selectedFile) {
    setStatus('Please upload an image first.');
    return;
  }

  isSearching = true;

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('limit', 1000); // Fetch 1000 items so we can load many pages
  
  // Log history
  if (typeof getResizedBase64 === 'function') {
      getResizedBase64(selectedFile).then(base64 => {
          let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];
          searchHistory.unshift({ type: 'image', query: selectedFile.name, imgData: base64, timestamp: new Date().toISOString() });
          if (searchHistory.length > 50) searchHistory = searchHistory.slice(0, 50);
          try {
              localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
          } catch(e) {
              console.warn("Storage full, clearing old history");
              searchHistory = searchHistory.slice(0, 10);
              localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
          }
      });
  }

  currentSearchType = 'image';
  setStatus('Searching...');
  clearResults();

  const startTime = performance.now();

  try {
    const response = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || 'Search failed.');
    }

    const data = await response.json();
    const endTime = performance.now();
    const searchTimeMs = data.search_time_ms || Math.round(endTime - startTime);

    // Track the uploaded image server URL for the compare feature.
    // This must be set whether or not an exact DB match was found so
    // that Image 1 in compare always shows the user's own upload.
    if (data.uploaded_file) {
        uploadedImageServerUrl = `/api/images/backend/uploads/${data.uploaded_file}`;
    } else {
        uploadedImageServerUrl = null;
    }

    // Exact match is only shown if the database returned a >=99% match
    if (data.has_exact_match) {
        renderExactMatchCard(selectedFile, data.exact_match_meta_category || data.predicted_category, data.uploaded_file, data.exact_match_meta_features);
    } else {
        renderExactMatchCard(null, null, null, null);
    }
    renderImageSearchResults(data.similar_results || [], data.other_model_results || [], searchTimeMs, data.predicted_category);
    setStatus(`Search complete.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    isSearching = false;
  }
}

async function runTextSearch() {
  const query = textSearchInput.value.trim();
  if (!query) {
    setStatus('Please enter a search query.');
    return;
  }

  const formData = new FormData();
  formData.append('query', query);
  formData.append('limit', 1000); 
  
  // Log history
  const searchTypeToLog = window._isVoiceSearch ? 'voice' : 'text';
  let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];
  searchHistory.unshift({ type: searchTypeToLog, query: query, timestamp: new Date().toISOString() });
  localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
  window._isVoiceSearch = false; // Reset

  currentSearchType = 'text';
  textSearchBtn.disabled = true;
  setStatus('Searching...');
  clearResults();
  
  const startTime = performance.now();

  try {
    const response = await fetch(`${API_BASE}/search_text`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || 'Text Search failed.');
    }

    const data = await response.json();
    const endTime = performance.now();
    const searchTimeMs = data.search_time_ms || Math.round(endTime - startTime);

    renderResults(data.results || [], searchTimeMs, data.category_filter || 'All Categories');
    setStatus(`Text search complete.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    textSearchBtn.disabled = false;
  }
}

// ─── Voice Search (Web Speech API — browser-native, no backend needed) ───────
let isRecording = false;
let recognition = null;
let silenceTimer = null;
let finalTranscript = '';
const SILENCE_TIMEOUT_MS = 2500; // wait 2.5s of silence before auto-stopping
const SPEECH_END_BUFFER_MS = 6000; // extra buffer after onspeechend fires mid-sentence

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

function setMicRecording(active) {
  isRecording = active;
  micBtn.classList.toggle('mic-active', active);
  micBtn.title = active ? 'Stop listening' : 'Search by Voice';
}

function stopVoice(triggerSearch = false) {
  clearTimeout(silenceTimer);
  if (recognition) {
    try { recognition.stop(); } catch (e) {}
  }
  setMicRecording(false);
  if (triggerSearch && textSearchInput.value.trim()) {
    setStatus('Voice recognised. Searching...');
    window._isVoiceSearch = true;
    const hasImage = !!selectedFile;
    if (hasImage) {
        runCombinedSearch();
    } else {
        runTextSearch();
    }
  } else if (!triggerSearch) {
    setStatus('Voice search cancelled.');
  }
}

function resetSilenceTimer() {
  clearTimeout(silenceTimer);
  silenceTimer = setTimeout(() => {
    stopVoice(true); // auto-trigger search after silence
  }, SILENCE_TIMEOUT_MS);
}

micBtn.addEventListener('click', () => {
  // If already recording, cancel it
  if (isRecording) {
    stopVoice(false);
    return;
  }

  if (!SpeechRecognition) {
    setStatus('❌ Voice search is not supported in this browser. Please use Chrome or Edge.');
    return;
  }

  const selectedLang = document.getElementById('voiceLangSelect')?.value || 'en-IN';

  recognition = new SpeechRecognition();
  recognition.lang = selectedLang;
  recognition.continuous = true;        // keep listening until we stop it
  recognition.interimResults = true;    // show words as they are spoken
  recognition.maxAlternatives = 1;

  finalTranscript = '';
  textSearchInput.value = '';
  textSearchInput.style.height = 'auto';

  recognition.onstart = () => {
    setMicRecording(true);
    setStatus('🎙️ Listening...');
    resetSilenceTimer(); // start initial silence guard
  };

  recognition.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += t + ' ';
      } else {
        interim += t;
      }
    }
    // Show live text in the search box
    textSearchInput.value = (finalTranscript + interim).trim();
    textSearchInput.style.height = 'auto';
    textSearchInput.style.height = textSearchInput.scrollHeight + 'px';
    // Reset the silence timer every time new words arrive
    resetSilenceTimer();
  };

  recognition.onspeechend = () => {
    // Browser detected a speech pause — do not stop immediately,
    // in case the user is just thinking between words/phrases.
    // Reset the silence timer to give them a full window to resume.
    resetSilenceTimer();
  };

  recognition.onerror = (event) => {
    setMicRecording(false);
    clearTimeout(silenceTimer);
    if (event.error === 'no-speech') {
      setStatus('⚠️ No speech detected. Please try again.');
    } else if (event.error === 'audio-capture') {
      setStatus('❌ Microphone not found. Please check your microphone.');
    } else if (event.error === 'not-allowed') {
      setStatus('❌ Microphone access denied. Please allow microphone in browser settings.');
    } else if (event.error === 'network') {
      setStatus('❌ Network error during voice recognition.');
    } else {
      setStatus(`❌ Voice error: ${event.error}. Please try again.`);
    }
  };

  recognition.onend = () => {
    // If still marked as recording, recognition ended unexpectedly — restart it
    // to keep continuous listening until silence is detected
    if (isRecording) {
      try { recognition.start(); } catch(e) { setMicRecording(false); }
    }
  };

  try {
    recognition.start();
  } catch (err) {
    setStatus(`❌ Could not start voice recognition: ${err.message}`);
  }
});

textSearchBtn.addEventListener('click', runTextSearch);
textSearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent default new line
        const hasImage = !!selectedFile;
        const hasText = !!textSearchInput.value.trim();
        if (hasImage && hasText) runCombinedSearch();
        else if (hasText) runTextSearch();
        else if (hasImage) runSearch();
    }
});

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) {
    selectedFile = null;
    clearUploadedPreview();
    setStatus('Select an image to begin.');
    return;
  }

  selectedFile = file;
  showUploadedPreview(file);
  setStatus(`Loaded ${file.name}`);
  runSearch();
});

searchBtn.addEventListener('click', () => {
    const hasImage = !!selectedFile;
    const hasText = !!textSearchInput.value.trim();
    if (hasImage && hasText) {
        runCombinedSearch();
    } else if (hasImage) {
        runSearch();
    } else if (hasText) {
        runTextSearch();
    } else {
        setStatus('Please upload an image or enter a search query.');
    }
});

const removeImageBtn = document.getElementById('removeImageBtn');
if (removeImageBtn) {
    removeImageBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectedFile = null;
        imageInput.value = '';
        clearUploadedPreview();
        setStatus('Image removed.');
    });
}

async function runCombinedSearch() {
  if (isSearching) return;
  isSearching = true;
  setStatus('Searching...');
  clearResults();
  
  const startTime = performance.now();
  let imageResults = [];
  let textResults = [];
  let imageCat = null;
  
  try {
    const formData1 = new FormData();
    formData1.append('image', selectedFile);
    formData1.append('limit', 1000);
    const res1 = await fetch(`${API_BASE}/search`, { method: 'POST', body: formData1 });
    if (res1.ok) {
       const data1 = await res1.json();
       // /api/search now returns similar_results/other_model_results
       // instead of a flat `results` array (see the plain image-search
       // path above); reconstruct the same combined pool this function
       // already expected.
       imageResults = (data1.similar_results || []).concat(data1.other_model_results || []);
       imageCat = data1.predicted_category;
    }
    
    const formData2 = new FormData();
    formData2.append('query', textSearchInput.value.trim());
    formData2.append('limit', 1000);
    const res2 = await fetch(`${API_BASE}/search_text`, { method: 'POST', body: formData2 });
    if (res2.ok) {
       const data2 = await res2.json();
       textResults = data2.results || [];
    }
    
    let combined = [];
    const textMap = new Map();
    textResults.forEach(r => textMap.set(r.image_url, r));
    
    imageResults.forEach(r => {
        if (textMap.has(r.image_url)) {
            const tr = textMap.get(r.image_url);
            combined.push({
                ...r,
                similarity: (r.similarity + tr.similarity) / 2
            });
        }
    });
    
    // Fallback to Union if intersection is empty
    if (combined.length === 0) {
        const unionMap = new Map();
        imageResults.forEach(r => unionMap.set(r.image_url, r));
        textResults.forEach(r => {
            if (!unionMap.has(r.image_url)) {
                unionMap.set(r.image_url, r);
            } else {
                let existing = unionMap.get(r.image_url);
                existing.similarity = Math.max(existing.similarity, r.similarity);
            }
        });
        combined = Array.from(unionMap.values());
    }
    
    combined.sort((a,b) => b.similarity - a.similarity);
    
    const endTime = performance.now();
    currentSearchType = 'image';
    renderResults(combined, Math.round(endTime - startTime), imageCat || 'Combined');
    setStatus(`Combined search complete.`);
  } catch (e) {
    setStatus(e.message);
  } finally {
    isSearching = false;
  }
}

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (event) => {
  const [file] = event.dataTransfer.files;
  if (!file || !file.type.startsWith('image/')) {
    setStatus('Please drop a valid image file.');
    return;
  }

  selectedFile = file;
  imageInput.files = event.dataTransfer.files;
  showUploadedPreview(file);
  setStatus(`Loaded ${file.name}`);
  runSearch();
});

function renderNextBatch() {
    if (!currentSimilarMatches || currentSimilarMatches.length === 0) return;
    
    const end = Math.min(currentSimilarIndex + ITEMS_PER_PAGE, currentSimilarMatches.length);
    for (let i = currentSimilarIndex; i < end; i++) {
        similarResultsList.appendChild(createResultCard(currentSimilarMatches[i]));
    }
    currentSimilarIndex = end;
    
    if (currentSimilarIndex < currentSimilarMatches.length) {
        if (loadMoreWrapper) loadMoreWrapper.style.display = 'flex';
        if (otherModelsTrigger) otherModelsTrigger.style.display = 'none';
    } else {
        if (loadMoreWrapper) loadMoreWrapper.style.display = 'none';
        if (otherModelsTrigger && currentOtherMatches.length > 0) {
            otherModelsTrigger.style.display = 'block';
        }
    }
}

function renderNextOtherBatch() {
    if (!currentOtherMatches || currentOtherMatches.length === 0) return;
    
    const end = Math.min(currentOtherIndex + ITEMS_PER_PAGE, currentOtherMatches.length);
    for (let i = currentOtherIndex; i < end; i++) {
        otherResultsList.appendChild(createResultCard(currentOtherMatches[i]));
    }
    currentOtherIndex = end;
    
    if (currentOtherIndex < currentOtherMatches.length) {
        if (loadMoreOtherWrapper) loadMoreOtherWrapper.style.display = 'flex';
    } else {
        if (loadMoreOtherWrapper) loadMoreOtherWrapper.style.display = 'none';
    }
}

if (loadMoreBtn) {
    loadMoreBtn.addEventListener('click', renderNextBatch);
}

if (showOtherModelsBtn) {
    showOtherModelsBtn.addEventListener('click', () => {
        if (otherModelsTrigger) otherModelsTrigger.style.display = 'none';
        if (otherModelsSection) {
            otherModelsSection.style.display = 'block';
            renderNextOtherBatch();
        }
    });
}

if (loadMoreOtherBtn) {
    loadMoreOtherBtn.addEventListener('click', renderNextOtherBatch);
}

// View Swapping Controls
window.showSearch = function() {
    const searchView = document.getElementById('searchView');
    const contentFrame = document.getElementById('contentFrame');
    if (searchView) searchView.style.display = 'block';
    if (contentFrame) {
        contentFrame.style.display = 'none';
        contentFrame.src = '';
    }
    // Update active state on side nav if needed
    document.querySelectorAll('.side-btn').forEach(btn => btn.classList.remove('active'));
    const dashBtn = document.querySelector('.dashboard-btn');
    if(dashBtn) dashBtn.classList.add('active');
};

window.showIframe = function(url) {
    const searchView = document.getElementById('searchView');
    const contentFrame = document.getElementById('contentFrame');
    if (searchView) searchView.style.display = 'none';
    if (contentFrame) {
        contentFrame.src = url;
        contentFrame.style.display = 'block';
    }
    
    document.querySelectorAll('.side-btn').forEach(btn => btn.classList.remove('active'));
    if (url.includes('/admin')) {
        const adminBtn = document.querySelector('.admin-panel-btn');
        if(adminBtn) adminBtn.classList.add('active');
    } else if (url.includes('/compare')) {
        const compBtn = document.querySelector('.compare-ai-btn');
        if(compBtn) compBtn.classList.add('active');
    }
};

function enterCompareMode() {
    compareModeActive = true;
    document.body.classList.add('compare-mode');
    const banner = document.getElementById('compareSelectionBanner');
    if (banner) banner.style.display = 'flex';
    setStatus('Select the second image to compare.');
}

function exitCompareMode() {
    compareModeActive = false;
    document.body.classList.remove('compare-mode');
    const banner = document.getElementById('compareSelectionBanner');
    if (banner) banner.style.display = 'none';
    document.querySelectorAll('.selected-for-compare').forEach(el => el.classList.remove('selected-for-compare'));
    compareImage2 = null;
    setStatus('Selection cancelled.');
}

function startAutomaticComparison() {
    // Always re-derive compareImage1 fresh so a prior search's image
    // never bleeds into the next comparison.
    compareImage1 = null;

    // Prefer the dedicated uploaded-image URL (set when image search completes).
    // Fall back to the exact-match card, then the first similar result.
    if (currentSearchType === 'image' && uploadedImageServerUrl) {
        compareImage1 = uploadedImageServerUrl;
    } else if (currentSearchType === 'image' && exactResultsList.firstChild) {
        const imgEl = exactResultsList.firstChild.querySelector('.result-image');
        compareImage1 = imgEl.dataset.serverUrl || imgEl.src;
    } else if (similarResultsList.firstChild) {
        const imgEl = similarResultsList.firstChild.querySelector('.result-image');
        compareImage1 = imgEl.dataset.serverUrl || imgEl.src;
    }
    
    if (!compareImage1) {
         setStatus('No first image found to compare against.');
         exitCompareMode();
         return;
    }
    
    const compareUrl = `/compare?img1=${encodeURIComponent(compareImage1)}&img2=${encodeURIComponent(compareImage2)}`;
    console.log('Opening compare in modal:', compareUrl);
    
    const compareModal = document.getElementById('compareModal');
    const compareFrame = document.getElementById('compareFrame');
    const compareLoading = document.getElementById('compareModalLoading');
    
    if (compareModal && compareFrame && compareLoading) {
        compareModal.style.display = 'flex';
        compareLoading.style.display = 'flex';
        compareFrame.style.display = 'none';
        
        compareFrame.onload = function() {
            compareLoading.style.display = 'none';
            compareFrame.style.display = 'block';
        };
        
        compareFrame.src = compareUrl;
    }
    
    exitCompareMode();
    setStatus('Comparison opened.');
}

window.closeCompareModal = function() {
    const compareModal = document.getElementById('compareModal');
    const compareFrame = document.getElementById('compareFrame');
    if (compareModal) compareModal.style.display = 'none';
    if (compareFrame) {
        compareFrame.src = '';
        compareFrame.style.display = 'none';
    }
};

const cancelCompareBtn = document.getElementById('cancelCompareBtn');
if (cancelCompareBtn) {
    cancelCompareBtn.addEventListener('click', exitCompareMode);
}

const compareBtn = document.getElementById('compareBtn');
if (compareBtn) {
    compareBtn.addEventListener('click', enterCompareMode);
}

clearUploadedPreview();
clearResults();

// --- Collections, History, and Navigation Additions ---

let myCollections = JSON.parse(localStorage.getItem('myCollections')) || [];
function updateCollectionCount() {
    const c1 = document.getElementById('sidebarCollectionCount');
    const c2 = document.getElementById('tabCollectionCount');
    if (c1) c1.textContent = myCollections.length;
    if (c2) c2.textContent = myCollections.length;
}
updateCollectionCount();

// Removed global heart listener since it's now bound directly in createResultCard

// Setup navigation shortcuts
const navImageSearch = document.getElementById('navImageSearch');
const navTextSearch = document.getElementById('navTextSearch');
const tabImageSearch = document.getElementById('tabImageSearch');
const tabTextSearch = document.getElementById('tabTextSearch');

[navImageSearch, tabImageSearch].forEach(el => {
    if (el) el.addEventListener('click', () => {
        showSearch();
        const dz = document.getElementById('dropZone');
        dz.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        dz.classList.add('blink-highlight');
        setTimeout(() => {
            dz.classList.remove('blink-highlight');
            document.getElementById('imageInput').click();
        }, 600);
    });
});

[navTextSearch, tabTextSearch].forEach(el => {
    if (el) el.addEventListener('click', () => {
        showSearch();
        const ti = document.getElementById('textSearchInput');
        ti.focus();
        ti.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
});

// Scroll to Top Button Logic
const scrollTopBtn = document.getElementById('scrollTopBtn');
const mainContentArea = document.getElementById('mainContentArea');

if (mainContentArea && scrollTopBtn) {
    mainContentArea.addEventListener('scroll', () => {
        if (mainContentArea.scrollTop > 300) {
            scrollTopBtn.classList.add('visible');
        } else {
            scrollTopBtn.classList.remove('visible');
        }
    });

    scrollTopBtn.addEventListener('click', () => {
        const target = document.getElementById('dropZone') || mainContentArea;
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
}

function getResizedBase64(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let w = img.width;
                let h = img.height;
                const MAX_DIM = 800;
                if (w > h && w > MAX_DIM) { h *= MAX_DIM / w; w = MAX_DIM; }
                else if (h > MAX_DIM) { w *= MAX_DIM / h; h = MAX_DIM; }
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);
                resolve(canvas.toDataURL('image/jpeg', 0.8));
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

window.searchFromImageUrl = async function(url) {
    try {
        setStatus('Fetching image for search...');
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch image');
        const blob = await response.blob();
        
        let filename = 'searched_image.jpg';
        try {
            const pathUrl = new URL(url);
            const pathParts = pathUrl.pathname.split('/');
            const lastPart = pathParts[pathParts.length - 1];
            if (lastPart) filename = lastPart;
        } catch(e) {}

        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
        
        if (typeof showSearch === 'function') showSearch();

        const dz = document.getElementById('dropZone');
        if (dz) {
            dz.scrollIntoView({ behavior: 'smooth', block: 'center' });
            dz.classList.add('blink-highlight');
            setTimeout(() => dz.classList.remove('blink-highlight'), 600);
        }
        
        selectedFile = file;
        showUploadedPreview(file);
        setStatus(`Loaded ${file.name}`);
        runSearch();
    } catch (err) {
        console.error('Error loading image for search:', err);
        setStatus('Error loading image for search.');
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // Mobile Hamburger Menu Logic
    const hamburgerBtn = document.getElementById('hamburgerBtn');
    if (hamburgerBtn) {
        hamburgerBtn.addEventListener('click', () => {
            const navSection = document.querySelector('.nav-section');
            const navCtas = document.querySelectorAll('.nav-cta');
            if (navSection) navSection.classList.toggle('expanded');
            navCtas.forEach(cta => cta.classList.toggle('expanded'));
        });
    }

    // --- Hover Preview Logic ---
    const uploadedPreviewChip = document.getElementById('uploadedPreviewChip');
    const hoverPreviewContainer = document.getElementById('hoverPreviewContainer');
    const hoverPreviewImage = document.getElementById('hoverPreviewImage');

    if (uploadedPreviewChip && hoverPreviewContainer && hoverPreviewImage) {
        uploadedPreviewChip.addEventListener('mouseenter', () => {
            if (selectedFile) {
                hoverPreviewImage.src = URL.createObjectURL(selectedFile);
                hoverPreviewContainer.classList.add('active');
            }
        });

        uploadedPreviewChip.addEventListener('mousemove', (e) => {
            if (hoverPreviewContainer.classList.contains('active')) {
                const x = e.clientX + 20; // 20px offset
                const y = e.clientY + 20;
                
                // Basic bounds checking so it doesn't go off screen
                const containerWidth = hoverPreviewContainer.offsetWidth || 400;
                const containerHeight = hoverPreviewContainer.offsetHeight || 400;
                
                let finalX = x;
                let finalY = y;

                if (x + containerWidth > window.innerWidth) {
                    finalX = e.clientX - containerWidth - 20;
                }
                if (y + containerHeight > window.innerHeight) {
                    finalY = e.clientY - containerHeight - 20;
                }

                hoverPreviewContainer.style.left = `${finalX}px`;
                hoverPreviewContainer.style.top = `${finalY}px`;
            }
        });

        uploadedPreviewChip.addEventListener('mouseleave', () => {
            hoverPreviewContainer.classList.remove('active');
            setTimeout(() => {
                if (!hoverPreviewContainer.classList.contains('active')) {
                    hoverPreviewImage.src = '';
                }
            }, 200);
        });
    }
});


// Scroll to Top Button Logic


// End of script.js
