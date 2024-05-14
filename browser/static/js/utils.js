let setTriggerValues = (individualParameters) => {
  var xi = individualParameters['xi'];
  document.getElementById('acc_factor').value = xi;

  var tau = individualParameters['tau'];
  document.getElementById('time_shift').value = tau;

  var sources = individualParameters['sources'];
  // for(var i=0; i<model['source_dimension']; ++i) {
  //  document.getElementById('geom_'+i).value = sources[i];
  // };

  var nemnem = individualParameters['nemnem'];
  document.getElementById('nemnem').value = nemnem;
}

let resetTriggerValues = () => {
  if(!model) {
    return;
  }
  var individualParameters = {
    'xi': 0,
    'tau': 0,
    'sources': new Array(model['source_dimension']).fill(0),
    'prs_cc': 0,
    'prs_dep': 0,
    'prs_diab': 0,
    'prs_hyp': 0,
    'prs_ldl': 0,
    'prs_ad': 0,
    'prs_hl': 0,
  }
  setTriggerValues(individualParameters);
  changeTriggerText(individualParameters);
}

let getTriggerValues = () => {
  var values = {
    'xi': parseFloat(document.getElementById('acc_factor').value),
    'tau': parseFloat(document.getElementById('time_shift').value),
    'prs_cc': parseFloat(document.getElementById('prs_cc').value),
    'prs_dep': parseFloat(document.getElementById('prs_dep').value),
    'prs_diab': parseFloat(document.getElementById('prs_diab').value),
    'prs_hyp': parseFloat(document.getElementById('prs_hyp').value),
    'prs_ldl': parseFloat(document.getElementById('prs_ldl').value),
    'prs_ad': parseFloat(document.getElementById('prs_ad').value),
    'prs_hl': parseFloat(document.getElementById('prs_hl').value),
    'sources': []
  }

  // for(var i=0; i<model['source_dimension']; ++i) {
  //  values['sources'].push(parseFloat(document.getElementById('geom_'+i).value));
  //}

  return values
}

let changeTriggerText = (indivParameters) => {
  // For the acceleration factor: we store & slide in log-space (xi) but we display exp(xi)
  var xi = indivParameters['xi'];
  document.getElementById('acc_factor').previousSibling.innerHTML = 'Acceleration factor: ' + Math.exp(xi).toFixed(DECIMALS_XI);

  var tau = indivParameters['tau'];
  document.getElementById('time_shift').previousSibling.innerHTML = 'Time shift: ' + tau.toFixed(DECIMALS_TAU);

  var sources = indivParameters['sources'];
  // for(var i=0; i<model['source_dimension']; ++i) {
  //  document.getElementById('geom_'+i).previousSibling.innerHTML = 'Geometric pattern ' + (i+1) + ': ' + sources[i].toFixed(DECIMALS_SOURCES);
  //}

  document.getElementById('prs_cc').previousSibling.innerHTML = 'PRS Colon Cancer' + ':' + indivParameters['prs_cc'].toFixed(1);
  document.getElementById('prs_dep').previousSibling.innerHTML = 'PRS Depression' + ':' + indivParameters['prs_dep'].toFixed(1);
  document.getElementById('prs_diab').previousSibling.innerHTML = 'PRS Diabetes' + ':' + indivParameters['prs_diab'].toFixed(1);
  document.getElementById('prs_hyp').previousSibling.innerHTML = 'PRS Hypertension' + ':' + indivParameters['prs_hyp'].toFixed(1);
  document.getElementById('prs_ldl').previousSibling.innerHTML = 'PRS LDL Cholesterol' + ':' + indivParameters['prs_ldl'].toFixed(1);
  document.getElementById('prs_ad').previousSibling.innerHTML = 'PRS AD' + ':' + indivParameters['prs_ad'].toFixed(1);
  document.getElementById('prs_hl').previousSibling.innerHTML = 'PRS Hearing Loss' + ':' + indivParameters['prs_hl'].toFixed(1);
  
}

let onTriggerChange = () => {
  var indivParameters = getTriggerValues();
  changeTriggerText(indivParameters);
  var values = compute_values(ages, model, indivParameters);

  for(var i=0; i<model['dimension']; ++i) {
    var data = convertData(ages, values[i])
    myChart.data.datasets[i].data = data;
  }
  myChart.update();
}

let convertData = (ages, values) => {
  var scatter = []
  for(var i=0; i<ages.length; i++){
    scatter.push({x:ages[i], y:values[i]})
  }
  return scatter;
}

let addRow = () => {
  hot.alter('insert_row');
}

let removeRow = () => {
  hot.alter('remove_row');
}

let argMax = (array) => {
  var greatest;
  var indexOfGreatest;
  for (var i = 0; i < array.length; i++) {
    if (!greatest || array[i] > greatest) {
      greatest = array[i];
      indexOfGreatest = i;
    }
  }
  return indexOfGreatest;
}
