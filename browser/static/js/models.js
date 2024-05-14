let compute_values = (ages, model_parameters, individual_parameters) => {
  if(model_parameters['name'] == 'logistic_parallel') {
    return compute_logistic_parallel(ages, model_parameters['parameters'], individual_parameters)
  } else if(model_parameters['name'] == 'logistic' || model_parameters['name'] == 'univariate_logistic') {
    return compute_logistic_covariates(ages, model_parameters['parameters'], individual_parameters, model_parameters)
  } else if(model_parameters['name'] == 'linear' || model_parameters['name'] == 'univariate_linear') {
    return compute_linear(ages, model_parameters['parameters'], individual_parameters)
  } else {
    alert("Unknown model: " + model_parameters['name'])
  }
}

let compute_linear = (ages, parameters, individual_parameters) => {
  // Model parameters
  var univariate = !('v0' in parameters)
  var t0 = parameters['tau_mean']
  var g = parameters['g']
  var log_v0 = null
  var mixing_matrix = null

  if (univariate) {
    if (typeof g == 'number'){ g = [g] }
    log_v0 = [parameters['xi_mean']]
  } else {
    log_v0 = parameters['v0']
    mixing_matrix = parameters['mixing_matrix']
  }

  // Individual parameters
  var alpha = Math.exp(individual_parameters['xi'])
  var tau = individual_parameters['tau']
  var sources = individual_parameters['sources']
  var space_shift = new Array(g.length).fill(0)
  if (mixing_matrix) {
    space_shift = math.multiply(mixing_matrix, sources);
  }

  // Compute values
  var outputs = []
  for(var i=0; i < g.length; ++i) {
    var output = []
    var v_i = Math.exp(log_v0[i])
    for(var j=0; j < ages.length; ++j) {
      var r_age = alpha * (ages[j] - t0 - tau);
      var val = g[i] + space_shift[i] + v_i * r_age;
      output.push(val);
    }
    outputs.push(output);
  }
  return outputs;
}

let get_deltas_i = (model, parameters, i) => {
  // For ordinal models: compute deltas for the feature `i`
  if ('deltas' in parameters) {
    // Batched deltas
    deltas_i = parameters['deltas'][i];
    // Truncate before infinity if any
    first_inf = deltas_i.indexOf(Infinity)
    if (first_inf != -1){
      deltas_i = deltas_i.slice(0, first_inf)
    }
  } else {
    deltas_i = parameters['deltas_'+model['features'][i]];
  }
  // Prepend 0, exponentialize (> 0) and cumsum the deltas
  deltas_i = [0].concat(math.cumsum(math.exp(deltas_i)))

  return deltas_i;
}

let compute_ordinal_pdf_from_ordinal_sf = (sf) => {
  // cf. leaspy.models.utils.ordinal.compute_ordinal_pdf_from_ordinal_sf
  var sf_sup = [1].concat(sf) // P(Y >= 0) = 1
  var sf_inf = sf.concat([0]) // P(Y > max_level) = 0
  return math.diff([sf_inf, sf_sup])[0] // = sf_sup - sf_inf
}

let compute_ordinal_expectation = (pdf) => {
  var cs = math.cumsum(pdf.slice(1).reverse())
  return math.sum(cs)
}

let compute_logistic = (ages, parameters, individual_parameters, model) => {
  // Specific types of models
  var univariate = !('v0' in parameters)
  var is_ordinal = 'noise_model' in model && model['noise_model'].startsWith('ordinal')

  // Model parameters
  var t0 = parameters['tau_mean']
  var log_g = parameters['g']
  var log_v0 = null
  var mixing_matrix = null

  if (univariate) {
    if (typeof log_g == 'number'){ log_g = [log_g] }
    log_v0 = [parameters['xi_mean']]
  } else {
    log_v0 = parameters['v0']
    mixing_matrix = parameters['mixing_matrix']
  }

  // Individual parameters
  var alpha = Math.exp(individual_parameters['xi'])
  var tau = individual_parameters['tau']
  var sources = individual_parameters['sources']
  var space_shift = new Array(log_g.length).fill(0)
  if (mixing_matrix) {
    space_shift = math.multiply(mixing_matrix, sources);
  }

  // Compute values
  var outputs = []
  for(var i=0; i < log_g.length; ++i) {

    var output = []
    var g_i = Math.exp(log_g[i])
    var v_i = Math.exp(log_v0[i])
    if (univariate) {
      var b_i = 1. // no such thing in univariate
    } else {
      var b_i = (1+g_i) * (1+g_i) / g_i
    }

    var deltas_i = null;

    if (is_ordinal) {
      deltas_i = get_deltas_i(model, parameters, i)
    }

    for(var j=0; j < ages.length; ++j) {
      var r_age = alpha * (ages[j] - t0 - tau);
      if (!is_ordinal) {
        var x_ij = v_i * r_age + space_shift[i];
        output.push(1./(1. + Math.exp(log_g[i] - b_i * x_ij)))
      } else {
        // Ordinal model --> loop on levels
        var sf_ij = [];
        for(var k=0; k < deltas_i.length; ++k) {
          // Survival function: P(Y_i > k), k=0..max_level_i-1
          var x_ijk;
          if (univariate) {
            // <!> This is subtle but by design: in univariate `deltas` <-> `deltas * v0` compared to multivariate
            x_ijk = v_i * r_age - deltas_i[k];
          } else {
            x_ijk = v_i * (r_age - deltas_i[k]) + space_shift[i];
          }
          sf_ij.push(1./(1. + Math.exp(log_g[i] - b_i * x_ijk)))
        }
        // Compute MLE (or expectation?)
        var pdf_ij = compute_ordinal_pdf_from_ordinal_sf(sf_ij);
        var ordinal_val_ij;

        if (ORDINAL_EXPECTATION) {
          ordinal_val_ij = compute_ordinal_expectation(pdf_ij);
        } else {
          ordinal_val_ij = argMax(pdf_ij); // MLE
        }
        if (ORDINAL_NORMALIZE) {
          ordinal_val_ij /= pdf_ij.length - 1; // max_level_ft <=> 1
        }
        output.push(ordinal_val_ij);
      }
    }
    outputs.push(output);
  }

  return outputs;
}

function transpose(matrix) {
  return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

let compute_logistic_covariates = (ages, parameters, individual_parameters, model) => {
  //Specific types of models /
  var univariate = false
  var is_ordinal = 'noise_model' in model && model['noise_model'].startsWith('ordinal')

  // Model parameters
  var t0 = parameters['link_t_mean']
  var log_g = parameters['link_g']
  var log_v0 = null
  var mixing_matrix = null

  log_v0 = parameters['link_v0']
  mixing_matrix = parameters['mixing_matrix']

  // Individual parameters
  var alpha = Math.exp(individual_parameters['xi'])
  var tau = individual_parameters['tau']
  var sources = individual_parameters['sources']
  //var cov_value = individual_parameters['nemnem']
  var cov_values = new Array(log_g[0].length).fill(0)
  cov_values[0] = individual_parameters['prs_cc']
  cov_values[1] = individual_parameters['prs_dep']
  cov_values[2] = individual_parameters['prs_diab']
  cov_values[3] = individual_parameters['prs_hyp']
  cov_values[4] = individual_parameters['prs_ldl']
  cov_values[5] = individual_parameters['prs_ad']
  cov_values[6] = individual_parameters['prs_hl']

  cov_values[7] = 1
  console.log(cov_values)
  var space_shift = new Array(log_g.length).fill(0)
  console.log(log_v0.length)
  console.log(log_v0[0].length)

  log_v0 = math.multiply(log_v0, cov_values)
  log_g = math.multiply(log_g, cov_values)
  t0 = math.multiply(t0, cov_values)

  if (mixing_matrix) {
    space_shift = math.multiply(mixing_matrix, sources);
  }

  // Compute values
  var outputs = []
  for(var i=0; i < log_g.length; ++i) {

    var output = []
    var g_i = Math.exp(log_g[i])
    var v_i = Math.exp(log_v0[i])//+ cov_value)
    if (univariate) {
      var b_i = 1. // no such thing in univariate
    } else {
      var b_i = (1+g_i) * (1+g_i) / g_i
    }

    var deltas_i = null;

    if (is_ordinal) {
      deltas_i = get_deltas_i(model, parameters, i)
    }

    for(var j=0; j < ages.length; ++j) {
      var r_age = alpha * (ages[j] - t0 - tau);
      if (!is_ordinal) {
        var x_ij = v_i * r_age + space_shift[i];
        output.push(1./(1. + Math.exp(log_g[i] - b_i * x_ij)))
      } else {
        // Ordinal model --> loop on levels
        var sf_ij = [];
        for(var k=0; k < deltas_i.length; ++k) {
          // Survival function: P(Y_i > k), k=0..max_level_i-1
          var x_ijk;
          if (univariate) {
            // <!> This is subtle but by design: in univariate `deltas` <-> `deltas * v0` compared to multivariate
            x_ijk = v_i * r_age - deltas_i[k];
          } else {
            x_ijk = v_i * (r_age - deltas_i[k]) + space_shift[i];
          }
          sf_ij.push(1./(1. + Math.exp(log_g[i] - b_i * x_ijk)))
        }
        // Compute MLE (or expectation?)
        var pdf_ij = compute_ordinal_pdf_from_ordinal_sf(sf_ij);
        var ordinal_val_ij;

        if (ORDINAL_EXPECTATION) {
          ordinal_val_ij = compute_ordinal_expectation(pdf_ij);
        } else {
          ordinal_val_ij = argMax(pdf_ij); // MLE
        }
        if (ORDINAL_NORMALIZE) {
          ordinal_val_ij /= pdf_ij.length - 1; // max_level_ft <=> 1
        }
        output.push(ordinal_val_ij);
      }
    }
    outputs.push(output);
  }

  return outputs;
}

let compute_logistic_parallel = (ages, parameters, individual_parameters) => {
  // Model parameters
  var log_g = parameters['g']
  var t0 = parameters['tau_mean']
  var v0 = Math.exp(parameters['xi_mean'])
  var deltas = [0].concat(parameters['deltas'])
  var mixing_matrix = parameters['mixing_matrix']

  // Individual parameters
  var alpha = Math.exp(individual_parameters['xi'])
  var tau = individual_parameters['tau']
  var sources = individual_parameters['sources']
  var space_shift = new Array(deltas.length).fill(0)
  if (mixing_matrix) {
    space_shift = math.multiply(mixing_matrix, sources);
  }

  // Compute values
  var outputs = []
  for(var i=0; i < deltas.length; ++i) {
    var output = []
    var g_i = Math.exp(log_g - deltas[i])
    var shift_i = deltas[i] + space_shift[i] * (1+g_i) * (1+g_i) / g_i
    for(var j=0; j < ages.length; ++j) {
      var r_age = v0 * alpha * (ages[j] - t0 - tau);
      var val = r_age + shift_i
      output.push(1./(1. + Math.exp(log_g - val)))
    }
    outputs.push(output)
  }
  return outputs
}
