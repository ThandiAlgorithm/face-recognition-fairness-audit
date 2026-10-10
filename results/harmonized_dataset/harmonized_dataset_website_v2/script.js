const labels = [
  ['White','white_preview.jpg'],['Black','black_preview.jpg'],['Indian','indian_preview.jpg'],
  ['East Asian','east_asian_preview.jpg'],['Southeast Asian','southeast_asian_preview.jpg'],
  ['Middle Eastern','middle_eastern_preview.jpg'],['Latino','latino_preview.jpg']
];

const grid = document.getElementById('galleryGrid');
labels.forEach(([label,file])=>{
  const card=document.createElement('article'); card.className='gallery-card';
  const img=document.createElement('img'); img.alt=label+' preview';
  img.src='data/project_outputs/FINAL_HARMONIZED_DATASET/previews/'+file;
  img.onerror=()=>{
    const box=document.createElement('div'); box.className='missing-preview';
    box.innerHTML='Preview not copied yet.<br>Run <code>.\\setup_all_data.ps1</code> in this website folder.';
    img.replaceWith(box)
  };
  const h=document.createElement('h3'); h.textContent=label;
  card.append(img,h); grid.append(card);
});

function parseCSV(text) {
  const rows=[]; let row=[]; let field=''; let quoted=false;
  for (let i=0;i<text.length;i++) {
    const c=text[i];
    if (quoted) {
      if (c==='"' && text[i+1]==='"') { field+='"'; i++; }
      else if (c==='"') quoted=false;
      else field+=c;
    } else {
      if (c==='"') quoted=true;
      else if (c===',') { row.push(field); field=''; }
      else if (c==='\n') { row.push(field.replace(/\r$/,'')); rows.push(row); row=[]; field=''; }
      else field+=c;
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

const manifestPath='data/project_outputs/FINAL_HARMONIZED_DATASET/harmonized_face_dataset_manifest.csv';
let manifestHeaders=[];
let manifestRows=[];
let filteredRows=[];
let page=0;
const pageSize=25;

const statusEl=document.getElementById('dataStatus');
const tableEl=document.getElementById('dataTable');
const filtersEl=document.getElementById('dataFilters');
const paginationEl=document.getElementById('pagination');

async function loadManifest() {
  if (!statusEl) return;
  statusEl.textContent='Loading 95,258-record manifest…';
  try {
    const r=await fetch(manifestPath);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const text=await r.text();
    const rows=parseCSV(text);
    manifestHeaders=rows[0];
    manifestRows=rows.slice(1).filter(r=>r.length && r.some(v=>v!==''));
    filteredRows=manifestRows;
    statusEl.innerHTML=`Loaded <strong>${manifestRows.length.toLocaleString()}</strong> records from the canonical manifest.`;
    buildFilters(); renderTable();
  } catch (e) {
    statusEl.innerHTML=`The full manifest has not been copied into the website yet. Run <code>.\\setup_all_data.ps1</code>, then start the site with <code>python -m http.server 8000</code>.`;
  }
}

function colIndex(name){ return manifestHeaders.indexOf(name); }
function uniq(name){ const i=colIndex(name); if(i<0) return []; return [...new Set(manifestRows.map(r=>r[i]).filter(Boolean))].sort(); }

function buildFilters(){
  filtersEl.innerHTML='';
  const defs=[['demographic_label','Label'],['source_dataset','Source'],['gender_label','Gender'],['ita_group','ITA']];
  defs.forEach(([col,label])=>{
    if(colIndex(col)<0) return;
    const wrap=document.createElement('label'); wrap.className='filter-control'; wrap.textContent=label+' ';
    const select=document.createElement('select'); select.dataset.col=col;
    select.innerHTML='<option value="">All</option>'+uniq(col).map(v=>`<option>${v}</option>`).join('');
    select.addEventListener('change', applyFilters); wrap.appendChild(select); filtersEl.appendChild(wrap);
  });
  const search=document.createElement('input'); search.type='search'; search.placeholder='Search any field…'; search.id='globalSearch';
  search.addEventListener('input', applyFilters); filtersEl.appendChild(search);
}

function applyFilters(){
  const selects=[...filtersEl.querySelectorAll('select')];
  const search=(document.getElementById('globalSearch')?.value||'').toLowerCase();
  filteredRows=manifestRows.filter(row=>{
    for(const s of selects){ const v=s.value; if(v && row[colIndex(s.dataset.col)]!==v) return false; }
    if(search && !row.join(' ').toLowerCase().includes(search)) return false;
    return true;
  });
  page=0; renderTable();
}

function renderTable(){
  if(!tableEl) return;
  const preferred=['record_id','source_dataset','source_image_id','demographic_label','source_demographic_label','gender_label','age_group','ita_group','identity_id'];
  const visible=preferred.filter(h=>colIndex(h)>=0);
  const start=page*pageSize; const end=Math.min(start+pageSize, filteredRows.length);
  tableEl.innerHTML='<thead><tr>'+visible.map(h=>`<th>${h}</th>`).join('')+'</tr></thead><tbody>'+filteredRows.slice(start,end).map(row=>'<tr>'+visible.map(h=>`<td>${escapeHtml(row[colIndex(h)]||'')}</td>`).join('')+'</tr>').join('')+'</tbody>';
  const totalPages=Math.max(1,Math.ceil(filteredRows.length/pageSize));
  paginationEl.innerHTML=`<button id="prevPage" ${page===0?'disabled':''}>Previous</button><span>${filteredRows.length.toLocaleString()} matches · page ${page+1} of ${totalPages}</span><button id="nextPage" ${page+1>=totalPages?'disabled':''}>Next</button>`;
  document.getElementById('prevPage')?.addEventListener('click',()=>{ if(page>0){page--;renderTable();} });
  document.getElementById('nextPage')?.addEventListener('click',()=>{ if(page+1<totalPages){page++;renderTable();} });
}

function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}

async function buildFileIndex(){
  const el=document.getElementById('allDataLinks'); if(!el) return;
  const links=[
    ['Canonical 95,258-record manifest','data/project_outputs/FINAL_HARMONIZED_DATASET/harmonized_face_dataset_manifest.csv'],
    ['Dataset card','data/project_outputs/FINAL_HARMONIZED_DATASET/DATASET_CARD.md'],
    ['Label schema','data/project_outputs/FINAL_HARMONIZED_DATASET/label_schema.json'],
    ['Demographic summary','data/project_outputs/FINAL_HARMONIZED_DATASET/demographic_summary.csv'],
    ['Source summary','data/project_outputs/FINAL_HARMONIZED_DATASET/source_summary.csv'],
    ['Gender summary','data/project_outputs/FINAL_HARMONIZED_DATASET/gender_summary.csv'],
    ['ITA summary','data/project_outputs/FINAL_HARMONIZED_DATASET/ita_summary.csv'],
    ['Initial label inventory folder','data/project_outputs/label_inventory/'],
    ['Candidate taxonomy v2 folder','data/project_outputs/candidate_taxonomy_v2/'],
    ['Balancing scenarios folder','data/project_outputs/balancing_scenarios/'],
    ['MVP volume-aware folder','data/project_outputs/mvp_volume_aware/'],
    ['Open Day population pack','data/project_outputs/open_day_pack/'],
    ['Final frozen training metadata folder','data/project_outputs/final_mvp_candidate/frozen_training/'],
    ['Final PDF report','downloads/Harmonized_Diversity_Aware_Face_Dataset_v1_Report.pdf'],
    ['Editable Word report','downloads/Harmonized_Diversity_Aware_Face_Dataset_v1_Report.docx']
  ];
  el.innerHTML=links.map(([name,path])=>`<a class="data-link" href="${path}" target="_blank">${name}</a>`).join('');
}

loadManifest(); buildFileIndex();
