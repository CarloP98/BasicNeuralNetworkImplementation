const operations = {
    sum: (a,b) => a+b,
    sub: (a,b) => a-b,
    mul: (a,b) => a*b,
    div: (a,b) => a/b,
    exp: (a) => Math.exp(a),
    log: (a) => Math.log(a),
    square: (a) => a**2
};

function transpose(m){
  return m[0].map((_, colIndex) => m.map(row => row[colIndex]));
}

function dotProd(m1, m2) {
  if(m1.constructor !== Array || m2.constructor !== Array){
    return ewop(m1, m2, "mul");
  }

    var result = [];
    for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m2[0].length; j++) {
            var sum = 0;
            for (var k = 0; k < m1[0].length; k++)
                sum += m1[i][k] * m2[k][j];
            result[i][j] = sum;
        }
    }
    return result;
}

function ewop(m1, m2, op) {
  s1 = matrixShape(m1);
  s2 = matrixShape(m2);

  if(s1[0]==undefined && s2[0]==undefined)
    return operations[op](m1,m2);
  if(s2[0]==undefined){
    var matrix = m1.map(function(arr){return arr.slice();});
    var scalar = m2;
    for(var row=0; row<matrix.length; row++)
      for(var col=0; col<matrix[0].length; col++)
        matrix[row][col] = operations[op](matrix[row][col], scalar);
    return matrix;
  }
  if(s1[0]==undefined){
    var matrix = m2.map(function(arr){return arr.slice();});
    var scalar = m1;
    for(var row=0; row<matrix.length; row++)
      for(var col=0; col<matrix[0].length; col++)
        matrix[row][col] = operations[op](scalar, matrix[row][col]);
    return matrix;
  }
  else{
    var result = m1.map(function(arr){return arr.slice();});
    for(var row=0; row<result.length; row++)
      for(var col=0; col<result[0].length; col++)
        result[row][col] = operations[op](result[row][col], m2[row][col]);
    return result;
  }
}

function ewop1(m1, op) {
  if(m1.constructor !== Array)
    return operations[op](m1);
  var matrix = m1.map(function(arr){return arr.slice();});
  for(var row=0; row<matrix.length; row++)
    for(var col=0; col<matrix[0].length; col++)
      matrix[row][col] = operations[op](matrix[row][col]);
  return matrix;
}

function matrixShape(m) {
    var dim = [];
    for (;;) {
        dim.push(m.length);
        if (Array.isArray(m[0]))
            m = m[0];
        else
            break;
    }
    return dim;
}

function msum(m){
  return m.reduce(function(a,b){return a.concat(b)}).reduce(function(a,b){return a+b});
}


function reshape(m, shape) {
  r = shape[0];
  c = shape[1];
   if (r * c !== m.length * m[0].length)
      return m
   const res = []
   let row = []
   m.forEach(items => items.forEach((num) => {
      row.push(num)
      if (row.length === c) {
         res.push(row)
         row = []
      }
   }))
   return res
};

function sum(m1, m2){
  return ewop(m1, m2, "sum")
}
function sub(m1, m2){
  return ewop(m1, m2, "sub")
}
function mul(m1, m2){
  return ewop(m1, m2, "mul")
}
function div(m1, m2){
  return ewop(m1, m2, "div")
}
function exp(m1){
  return ewop1(m1, "exp")
}
function log(m1){
  return ewop1(m1, "log")
}
function square(m1){
  return ewop1(m1, "square")
}
