function gaussianRandom() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

const activations = {
  	sigmoid: (x) => div(1, sum(1,exp(mul(-1,x)))),
  	relu: (x) => x.map(function(r){return r.map(function(c){return Math.max(0,c);});})
};

const activations_bp = {
	sigmoid: (dA, Z) => dotProd(dA,dotProd(activations["sigmoid"](Z), sub(1,activations["sigmoid"](Z)))),
	relu: (dA, Z) => dA.map((row,i)=>row.map((col,j)=>(Z[i][j]<=0)?0:col))
}

const costs ={
	"cross-entrophy": (AL,Y)=> (-1/Y.length)*msum(sum(dotProd(Y, log(AL)), dotProd(sub(1,Y), log(sub(1,AL))))),
	"mse": (AL,Y)=> (1/Y.length)*msum(square(sub(Y,AL))),
}

const costs_bp = {
	"cross-entrophy": (AL,Y) => mul(-1,sub(div(Y, AL),div(sub(1,Y),sub(1,AL)))),
	"mse": (AL,Y) => mul(-2/AL[0].length, sub(Y,AL))
}

const weightInitialize = (size) =>{
    return gaussianRandom()*Math.sqrt(2/size);
};


//EX
//classification
//var model = new nn([[5], [4, "sigmoid"],  [1, "sigmoid"]], "cross-entrophy", 0.001);
//regression
////var model = new nn([[5], [4, "sigmoid"], , [3, "relu"], [2]], "mse", 0.001);
class nn {
	constructor(nnShape, costf, learningRate) {
  		this.cache = [];
  		this.costf = costf;
		this.layers = nnShape;
		this.learningRate = learningRate;
  		this.parameters = this.initializeParameters();
	};

	train(xs, ys, iterations=1){
		for(var i=0; i<iterations; i++){
        	var prediction = this.modelForward(xs)
        	var cost = this.computeCost(prediction, ys)
        	//console.log(cost)
        	var grads = this.modelBackward(prediction, ys)
        	this.updateParameters(grads)
    	}
	}

	predict(xs){
		return this.modelForward(xs)
	}

  	initializeParameters(nnShape){
  		var parameters = {};
  		for(var layer=1; layer < this.layers.length; layer++){
  			parameters['W' + layer] = Array.from(Array(this.layers[layer][0]), () => Array.from(Array(this.layers[layer-1][0])).map(x=>weightInitialize(this.layers[layer-1][0])))
        	parameters['b' + layer] = Array(this.layers[layer][0]).fill(Array(1).fill(0));
  		}
    	return parameters
  	}

  	modelForward(X){
  		this.cache = []
  		var prevLayer = transpose(X);
	    for(var layer=1; layer < Math.floor(Object.keys(this.parameters).length/2)+1; layer++){
			var W = this.parameters['W' + layer]
			var b = this.parameters['b' + layer]
			var Z = dotProd(W, prevLayer)
			var Z = Z.map((row,i)=>row.map((col)=>col+parseFloat(b[i])))
			this.cache.push([prevLayer, W, b, Z])
			prevLayer = (this.layers[layer].length < 2)?Z:activations[this.layers[layer][1]](Z);
	    }
	    return prevLayer
  	}

  	modelBackward(AL, Y){
  		var grads = {}
    	var m = AL[0].length
      Y = reshape(Y, [AL.length, AL[0].length])
    	var dA_prev = costs_bp[this.costf](AL,Y)

    	for(var layer=this.cache.length; layer>0; layer--){
    		var [A_prev, W, b, Z] = this.cache[layer-1];
    		var dZ = (this.layers[layer].length < 2)?dA_prev:activations_bp[this.layers[layer][1]](dA_prev, Z)
    		var dW = dotProd(1/m, dotProd(dZ, transpose(A_prev)));
			var db = dotProd(1/m, dZ.map(r => [r.reduce((a, b) => a + b)]));
    		var dA_prev = dotProd(transpose(W), dZ);
			grads["dA" + (layer)] = dA_prev;
        	grads["dW" + (layer)] = dW;
        	grads["db" + (layer)] = db;
    	}
    	return grads
  	}

  	computeCost(AL, Y){
  		return costs[this.costf](transpose(AL),Y);
  	}

  	updateParameters(grads){
  		for(var layer=1; layer<Math.floor(Object.keys(this.parameters).length/2)+1; layer++){
  			this.parameters["W" + layer] = sub(this.parameters["W" + layer], dotProd(this.learningRate, grads["dW" + layer]));
  			this.parameters["b" + layer] = sub(this.parameters["b" + layer], dotProd(this.learningRate, grads["db" + layer]));
  		}
  	}
}
