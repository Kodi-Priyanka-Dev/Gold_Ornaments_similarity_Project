document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    let img1Path = urlParams.get('img1');
    let img2Path = urlParams.get('img2');
    
    if(!img1Path || !img2Path) {
        showError("Missing images for comparison.");
        return;
    }

    // Set images in UI with cache-busting query param to prevent stale browser cache
    // when the same filename is re-uploaded with different content
    const cacheBuster = 't=' + Date.now();
    const img1Src = img1Path + (img1Path.includes('?') ? '&' : '?') + cacheBuster;
    const img2Src = img2Path + (img2Path.includes('?') ? '&' : '?') + cacheBuster;
    document.getElementById('img1-display').src = img1Src;
    document.getElementById('img2-display').src = img2Src;

    // Call backend API
    fetch('/api/compare', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ img1: img1Path, img2: img2Path })
    })
    .then(response => response.json())
    .then(data => {
        if(data.error) {
            showError(data.error);
        } else {
            renderResults(data);
        }
    })
    .catch(error => {
        showError("Network error occurred during comparison.");
        console.error(error);
    });
});

function showError(msg) {
    document.getElementById('loadingState').style.display = 'none';
    const errorState = document.getElementById('errorState');
    errorState.style.display = 'block';
    document.getElementById('errorText').innerText = msg;
}

function renderResults(data) {
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('resultsContent').style.display = 'block';
    
    document.getElementById('matchScore').innerText = (data.match_score || '--') + '%';
    document.getElementById('summaryText').innerText = data.summary || '';
    
    document.getElementById('img1-info').innerText = data.image1_info || '';
    document.getElementById('img2-info').innerText = data.image2_info || '';
    
    const tbody = document.getElementById('attrBody');
    let html = '';
    
    // Extract category from the generated info string
    const cat1 = (data.image1_info || '').replace('Analysis -> Detected Category: ', '');
    const cat2 = (data.image2_info || '').replace('Analysis -> Detected Category: ', '');
    const catDiffers = cat1 !== cat2;
    
    // Category comparison
    html += `<tr class="${catDiffers ? 'differs' : ''}">
        <td class="label">Predicted Category</td>
        <td class="diff a-val">${cat1}${catDiffers ? '<span class="diff-flag">Differs</span>' : ''}</td>
        <td class="diff b-val">${cat2}</td>
    </tr>`;
    
    // Stone Type comparison
    if (data.attributes) {
        const stone1 = data.attributes.image1?.stone_type || 'Unknown';
        const stone2 = data.attributes.image2?.stone_type || 'Unknown';
        const stoneDiffers = stone1 !== stone2;
        const allStones1 = data.attributes.image1?.all_stones?.join(', ') || stone1;
        const allStones2 = data.attributes.image2?.all_stones?.join(', ') || stone2;
        
        html += `<tr class="${stoneDiffers ? 'differs' : ''}">
            <td class="label">Stone Type</td>
            <td class="diff a-val">${allStones1}${stoneDiffers ? '<span class="diff-flag">Differs</span>' : ''}</td>
            <td class="diff b-val">${allStones2}</td>
        </tr>`;
        
        // Shape comparison
        const shape1 = data.attributes.image1?.shape || 'Unknown';
        const shape2 = data.attributes.image2?.shape || 'Unknown';
        const shapeDiffers = shape1 !== shape2;
        const allShapes1 = data.attributes.image1?.all_shapes?.join(', ') || shape1;
        const allShapes2 = data.attributes.image2?.all_shapes?.join(', ') || shape2;
        
        html += `<tr class="${shapeDiffers ? 'differs' : ''}">
            <td class="label">Shape</td>
            <td class="diff a-val">${allShapes1}${shapeDiffers ? '<span class="diff-flag">Differs</span>' : ''}</td>
            <td class="diff b-val">${allShapes2}</td>
        </tr>`;
    }
    
    // Differences
    if (data.differences && data.differences.length > 0) {
        html += `<tr class="differs">
            <td class="label">Key Differences</td>
            <td class="diff a-val" colspan="2" style="white-space: pre-line;">${data.differences.join('\n')}<span class="diff-flag">Differs</span></td>
        </tr>`;
    }
    
    // Similarities
    if (data.similarities && data.similarities.length > 0) {
        html += `<tr>
            <td class="label">Similarities</td>
            <td class="diff a-val" colspan="2" style="white-space: pre-line; color: #bfe9dc;">${data.similarities.join('\n')}</td>
        </tr>`;
    }
    
    tbody.innerHTML = html;
    
    // Display comprehensive features
    if (data.comprehensive_features) {
        renderComprehensiveFeatures(data.comprehensive_features);
    }
}

function renderComprehensiveFeatures(features) {
    const container = document.getElementById('featuresContainer');
    if (!container) return;
    
    let html = '';
    
    // Helper function to format confidence percentage (ensures 100% displays correctly)
    const formatConfidence = (score) => {
        const percentage = Math.round(score * 100);
        return `${Math.min(percentage, 100)}%`; // Cap at 100% and always show %
    };
    
    // Helper function to render feature category
    const renderFeatureCategory = (title, features1, features2, categoryKey) => {
        if ((!features1 || features1.length === 0) && (!features2 || features2.length === 0)) {
            return '';
        }
        
        const f1Display = features1 && features1.length > 0 
            ? features1.map(f => `${f[0]} <span class="score">(${formatConfidence(f[1])})</span>`).join(', ')
            : 'Not detected';
        
        const f2Display = features2 && features2.length > 0
            ? features2.map(f => `${f[0]} <span class="score">(${formatConfidence(f[1])})</span>`).join(', ')
            : 'Not detected';
        
        const areSame = JSON.stringify(features1) === JSON.stringify(features2);
        
        return `
        <div class="feature-row ${areSame ? '' : 'differs'}">
            <div class="feature-label">${title}</div>
            <div class="feature-values">
                <div class="value img1">
                    ${f1Display}
                    ${!areSame && (features1 && features1.length > 0) ? '<span class="diff-indicator">●</span>' : ''}
                </div>
                <div class="value img2">
                    ${f2Display}
                    ${!areSame && (features2 && features2.length > 0) ? '<span class="diff-indicator">●</span>' : ''}
                </div>
            </div>
        </div>`;
    };
    
    // Metadata Features
    const meta1 = features.image1?.metadata_features || [];
    const meta2 = features.image2?.metadata_features || [];
    // Always show catalog features section
    const price1 = features.image1?.metadata_price ? '₹' + features.image1.metadata_price.toLocaleString('en-IN') : 'N/A';
    const price2 = features.image2?.metadata_price ? '₹' + features.image2.metadata_price.toLocaleString('en-IN') : 'N/A';
    const priceDiffers = price1 !== price2;
    
    html += `<div class="feature-section">
        <h3 class="section-title">📋 Catalog Features</h3>
        <div class="feature-row">
            <div class="feature-label">Labeled Features</div>
            <div class="feature-values">
                <div class="value img1">${meta1.length > 0 ? meta1.join(', ') : 'Not cataloged'}</div>
                <div class="value img2">${meta2.length > 0 ? meta2.join(', ') : 'Not cataloged'}</div>
            </div>
        </div>
        <div class="feature-row ${priceDiffers ? 'differs' : ''}">
            <div class="feature-label">Price</div>
            <div class="feature-values">
                <div class="value img1">${price1}${priceDiffers && price1 !== 'N/A' ? '<span class="diff-indicator">●</span>' : ''}</div>
                <div class="value img2">${price2}${priceDiffers && price2 !== 'N/A' ? '<span class="diff-indicator">●</span>' : ''}</div>
            </div>
        </div>
    </div>`;
    
    // Stone Features
    html += `<div class="feature-section">
        <h3 class="section-title">💎 Stone Features</h3>
        ${renderFeatureCategory('Stone Type', features.image1?.stone_features, features.image2?.stone_features, 'stones')}
    </div>`;
    
    // Design Features
    html += `<div class="feature-section">
        <h3 class="section-title">🎨 Design Features</h3>
        ${renderFeatureCategory('Design Style', features.image1?.design_features, features.image2?.design_features, 'design')}
    </div>`;
    
    // Color Features
    html += `<div class="feature-section">
        <h3 class="section-title">🌟 Color & Finish</h3>
        ${renderFeatureCategory('Color & Polish', features.image1?.color_features, features.image2?.color_features, 'color')}
    </div>`;
    
    // Craftsmanship Features
    html += `<div class="feature-section">
        <h3 class="section-title">✨ Craftsmanship</h3>
        ${renderFeatureCategory('Quality & Detail', features.image1?.craftsmanship_features, features.image2?.craftsmanship_features, 'craftsmanship')}
    </div>`;
    
    // Style Features
    html += `<div class="feature-section">
        <h3 class="section-title">👗 Style Category</h3>
        ${renderFeatureCategory('Style Type', features.image1?.style_features, features.image2?.style_features, 'style')}
    </div>`;
    
    // Weight Features
    html += `<div class="feature-section">
        <h3 class="section-title">⚖️ Weight Appearance</h3>
        ${renderFeatureCategory('Weight Impression', features.image1?.weight_features, features.image2?.weight_features, 'weight')}
    </div>`;
    
    // Full Metadata Raw Details
    const rawMeta1 = features.image1?.raw_metadata || {};
    const rawMeta2 = features.image2?.raw_metadata || {};
    const allKeys = new Set([...Object.keys(rawMeta1), ...Object.keys(rawMeta2)]);
    
    if (allKeys.size > 0) {
        html += `<div class="feature-section">
            <h3 class="section-title">🗄️ Full Metadata Records</h3>`;
            
        const ignoreKeys = ['features', 'size_type', 'split', 'confidence_score'];
        allKeys.forEach(key => {
            if (ignoreKeys.includes(key.toLowerCase())) return; // hide internal or unnecessary keys
            
            let val1 = rawMeta1[key];
            let val2 = rawMeta2[key];
            
            const displayVal1 = val1 !== undefined && val1 !== null ? String(val1) : '—';
            const displayVal2 = val2 !== undefined && val2 !== null ? String(val2) : '—';
            
            const keyFormatted = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            const differs = displayVal1 !== displayVal2;
            
            html += `<div class="feature-row ${differs ? 'differs' : ''}">
                <div class="feature-label">${keyFormatted}</div>
                <div class="feature-values">
                    <div class="value img1">${displayVal1}${differs && displayVal1 !== '—' ? '<span class="diff-indicator">●</span>' : ''}</div>
                    <div class="value img2">${displayVal2}${differs && displayVal2 !== '—' ? '<span class="diff-indicator">●</span>' : ''}</div>
                </div>
            </div>`;
        });
        
        html += `</div>`;
    }
    
    container.innerHTML = html;
}
