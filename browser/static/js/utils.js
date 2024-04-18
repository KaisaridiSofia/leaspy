let setTriggerValues = (individualParameters) => {
  var xi = individualParameters['xi'];
  document.getElementById('acc_factor').value = xi;

  var tau = individualParameters['tau'];
  document.getElementById('time_shift').value = tau;

  var sources = individualParameters['sources'];
  for(var i=0; i<model['source_dimension']; ++i) {
    document.getElementById('geom_'+i).value = sources[i];
  };

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
    'nemnem': 0,
  }
  setTriggerValues(individualParameters);
  changeTriggerText(individualParameters);
}

let getTriggerValues = () => {
  var values = {
    'xi': parseFloat(document.getElementById('acc_factor').value),
    'tau': parseFloat(document.getElementById('time_shift').value),
    'nemnem': parseFloat(document.getElementById('nemnem').value),
    'sources': []
  }

  for(var i=0; i<model['source_dimension']; ++i) {
    values['sources'].push(parseFloat(document.getElementById('geom_'+i).value));
  }

  return values
}

let changeTriggerText = (indivParameters) => {
  // For the acceleration factor: we store & slide in log-space (xi) but we display exp(xi)
  var xi = indivParameters['xi'];
  document.getElementById('acc_factor').previousSibling.innerHTML = 'Acceleration factor: ' + Math.exp(xi).toFixed(DECIMALS_XI);

  var tau = indivParameters['tau'];
  document.getElementById('time_shift').previousSibling.innerHTML = 'Time shift: ' + tau.toFixed(DECIMALS_TAU);

  var sources = indivParameters['sources'];
  for(var i=0; i<model['source_dimension']; ++i) {
    document.getElementById('geom_'+i).previousSibling.innerHTML = 'Geometric pattern ' + (i+1) + ': ' + sources[i].toFixed(DECIMALS_SOURCES);
  }

  document.getElementById('nemnem').previousSibling.innerHTML = 'Covariate ' + ':' + indivParameters['nemnem'].toFixed(1);
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
