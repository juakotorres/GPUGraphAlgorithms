(function(){'use strict';
var e="object"===typeof __ScalaJSEnv&&__ScalaJSEnv?__ScalaJSEnv:{},k="object"===typeof e.global&&e.global?e.global:"object"===typeof global&&global&&global.Object===Object?global:this;e.global=k;var m="object"===typeof e.exportsNamespace&&e.exportsNamespace?e.exportsNamespace:k;e.exportsNamespace=m;k.Object.freeze(e);var n={envInfo:e,semantics:{asInstanceOfs:2,arrayIndexOutOfBounds:2,moduleInit:2,strictFloats:!1,productionMode:!0},assumingES6:!1,linkerVersion:"0.6.21",globalThis:this};k.Object.freeze(n);
k.Object.freeze(n.semantics);var p=k.Math.imul||function(a,c){var b=a&65535,d=c&65535;return b*d+((a>>>16&65535)*d+b*(c>>>16&65535)<<16>>>0)|0},q=k.Math.clz32||function(a){if(0===a)return 32;var c=1;0===(a&4294901760)&&(a<<=16,c+=16);0===(a&4278190080)&&(a<<=8,c+=8);0===(a&4026531840)&&(a<<=4,c+=4);0===(a&3221225472)&&(a<<=2,c+=2);return c+(a>>31)},r=0,u=k.WeakMap?new k.WeakMap:null;function v(a){return function(c,b){return!(!c||!c.$classData||c.$classData.l!==b||c.$classData.j!==a)}}
function aa(a){for(var c in a)return c}function w(a,c,b){var d=new a.x(c[b]);if(b<c.length-1){a=a.m;b+=1;for(var f=d.B,g=0;g<f.length;g++)f[g]=w(a,c,b)}return d}function ba(a){switch(typeof a){case "string":return x(y);case "number":var c=a|0;return c===a?z(c)?x(A):B(c)?x(C):x(D):"number"===typeof a?x(E):x(ca);case "boolean":return x(da);case "undefined":return x(ea);default:return null===a?a.V():a&&a.$classData&&a.$classData.f.z?x(fa):a&&a.$classData?x(a.$classData):null}}
function ga(a){switch(typeof a){case "string":ha||(ha=(new F).d());for(var c=0,b=1,d=-1+(a.length|0)|0;0<=d;)c=c+p(65535&(a.charCodeAt(d)|0),b)|0,b=p(31,b),d=-1+d|0;return c;case "number":G||(G=(new H).d());c=G;b=a|0;if(b===a&&-Infinity!==1/a)c=b;else{if(c.g)c.D[0]=a,c=I(c.s[c.F]|0,c.s[c.E]|0);else{if(a!==a)c=!1,a=2047,b=+k.Math.pow(2,51);else if(Infinity===a||-Infinity===a)c=0>a,a=2047,b=0;else if(0===a)c=-Infinity===1/a,b=a=0;else if(d=(c=0>a)?-a:a,d>=+k.Math.pow(2,-1022)){a=+k.Math.pow(2,52);var b=
+k.Math.log(d)/.6931471805599453,b=+k.Math.floor(b)|0,b=1023>b?b:1023,f=+k.Math.pow(2,b);f>d&&(b=-1+b|0,f/=2);f=d/f*a;d=+k.Math.floor(f);f-=d;d=.5>f?d:.5<f?1+d:0!==d%2?1+d:d;2<=d/a&&(b=1+b|0,d=1);1023<b?(b=2047,d=0):(b=1023+b|0,d-=a);a=b;b=d}else a=d/+k.Math.pow(2,-1074),b=+k.Math.floor(a),d=a-b,a=0,b=.5>d?b:.5<d?1+b:0!==b%2?1+b:b;b=+b;c=I(b|0,(c?-2147483648:0)|(a|0)<<20|b/4294967296|0)}c=c.t^c.r}return c;case "boolean":return a?1231:1237;case "undefined":return 0;default:return a&&a.$classData||
null===a?a.y():null===u?42:ja(a)}}function ka(a,c){var b=k.Object.getPrototypeOf,d=k.Object.getOwnPropertyDescriptor;for(a=b(a);null!==a;){var f=d(a,c);if(void 0!==f)return f;a=b(a)}}function la(a,c,b){a=ka(a,b);if(void 0!==a)return b=a.get,void 0!==b?b.call(c):a.value}function ma(a,c,b,d){a=ka(a,b);if(void 0!==a&&(a=a.set,void 0!==a)){a.call(c,d);return}throw new k.TypeError("super has no setter '"+b+"'.");}
var ja=null!==u?function(a){switch(typeof a){case "string":case "number":case "boolean":case "undefined":return ga(a);default:if(null===a)return 0;var c=u.get(a);void 0===c&&(r=c=r+1|0,u.set(a,c));return c}}:function(a){if(a&&a.$classData){var c=a.$idHashCode$0;if(void 0!==c)return c;if(k.Object.isSealed(a))return 42;r=c=r+1|0;return a.$idHashCode$0=c}return null===a?0:ga(a)};function z(a){return"number"===typeof a&&a<<24>>24===a&&1/a!==1/-0}
function B(a){return"number"===typeof a&&a<<16>>16===a&&1/a!==1/-0}function J(){this.u=this.x=void 0;this.j=this.m=this.f=null;this.l=0;this.C=null;this.q="";this.b=this.o=this.p=void 0;this.name="";this.isRawJSType=this.isArrayClass=this.isInterface=this.isPrimitive=!1;this.isInstance=void 0}function K(a,c,b){var d=new J;d.f={};d.m=null;d.C=a;d.q=c;d.b=function(){return!1};d.name=b;d.isPrimitive=!0;d.isInstance=function(){return!1};return d}
function L(a,c,b,d,f,g,l){var h=new J,t=aa(a);g=g||function(a){return!!(a&&a.$classData&&a.$classData.f[t])};l=l||function(a,b){return!!(a&&a.$classData&&a.$classData.l===b&&a.$classData.j.f[t])};h.u=f;h.f=b;h.q="L"+c+";";h.b=l;h.name=c;h.isInterface=!1;h.isRawJSType=!!d;h.isInstance=g;return h}
function na(a){function c(a){if("number"===typeof a){this.B=Array(a);for(var b=0;b<a;b++)this.B[b]=f}else this.B=a}var b=new J,d=a.C,f="longZero"==d?M().v:d;c.prototype=new N;c.prototype.constructor=c;c.prototype.$classData=b;var d="["+a.q,g=a.j||a,l=a.l+1;b.x=c;b.u=oa;b.f={a:1,X:1,c:1};b.m=a;b.j=g;b.l=l;b.C=null;b.q=d;b.p=void 0;b.o=void 0;b.b=void 0;b.name=d;b.isPrimitive=!1;b.isInterface=!1;b.isArrayClass=!0;b.isInstance=function(a){return g.b(a,l)};return b}
function x(a){if(!a.p){var c=new O;c.n=a;a.p=c}return a.p}function pa(a){a.o||(a.o=na(a));return a.o}J.prototype.getFakeInstance=function(){return this===y?"some string":this===da?!1:this===A||this===C||this===D||this===E||this===ca?0:this===fa?M().v:this===ea?void 0:{$classData:this}};J.prototype.getSuperclass=function(){return this.u?x(this.u):null};J.prototype.getComponentType=function(){return this.m?x(this.m):null};
J.prototype.newArrayOfThisClass=function(a){for(var c=this,b=0;b<a.length;b++)c=pa(c);return w(c,a,0)};var qa=K(!1,"Z","boolean"),ra=K(0,"C","char"),sa=K(0,"B","byte"),ta=K(0,"S","short"),ua=K(0,"I","int"),va=K("longZero","J","long"),wa=K(0,"F","float"),xa=K(0,"D","double");qa.b=v(qa);ra.b=v(ra);sa.b=v(sa);ta.b=v(ta);ua.b=v(ua);va.b=v(va);wa.b=v(wa);xa.b=v(xa);function P(){}function N(){}N.prototype=P.prototype;P.prototype.d=function(){return this};P.prototype.A=function(){var a=ba(this).n.name,c=(+(this.y()>>>0)).toString(16);return a+"@"+c};P.prototype.y=function(){return ja(this)};P.prototype.toString=function(){return this.A()};var oa=L({a:0},"java.lang.Object",{a:1},void 0,void 0,function(a){return null!==a},function(a,c){if(a=a&&a.$classData){var b=a.l||0;return!(b<c)&&(b>c||!a.j.isPrimitive)}return!1});P.prototype.$classData=oa;
function Q(){this.h=null}Q.prototype=new N;Q.prototype.constructor=Q;Q.prototype.d=function(){R=this;this.h=k.jQuery;return this};Q.prototype.$classData=L({G:0},"org.scalajs.jquery.package$",{G:1,a:1});var R=void 0;function S(){R||(R=(new Q).d());return R}function T(){}T.prototype=new N;T.prototype.constructor=T;T.prototype.d=function(){return this};
function ya(){var a=U();(0,S().h)('\x3cbutton type\x3d"button"\x3eClick me!\x3c/button\x3e').click(function(){return function(){Aa()}}(a)).appendTo((0,S().h)("body"));(0,S().h)("body").append("\x3cp\x3eHello World\x3c/p\x3e")}function Aa(){U();(0,S().h)("body").append("\x3cp\x3e Holi bot\u00f3n apreta3 \x3c/p\x3e")}T.prototype.$classData=L({H:0},"tutorial.webapp.TutorialApp$",{H:1,a:1});var V=void 0;function U(){V||(V=(new T).d());return V}function O(){this.n=null}O.prototype=new N;
O.prototype.constructor=O;O.prototype.A=function(){return(this.n.isInterface?"interface ":this.n.isPrimitive?"":"class ")+this.n.name};O.prototype.$classData=L({L:0},"java.lang.Class",{L:1,a:1});function H(){this.g=!1;this.D=this.s=this.k=null;this.w=!1;this.F=this.E=0}H.prototype=new N;H.prototype.constructor=H;
H.prototype.d=function(){G=this;this.k=(this.g=!!(k.ArrayBuffer&&k.Int32Array&&k.Float32Array&&k.Float64Array))?new k.ArrayBuffer(8):null;this.s=this.g?new k.Int32Array(this.k,0,2):null;this.g&&new k.Float32Array(this.k,0,2);this.D=this.g?new k.Float64Array(this.k,0,1):null;if(this.g)this.s[0]=16909060,a=1===((new k.Int8Array(this.k,0,8))[0]|0);else var a=!0;this.E=(this.w=a)?0:1;this.F=this.w?1:0;return this};H.prototype.$classData=L({R:0},"scala.scalajs.runtime.Bits$",{R:1,a:1});var G=void 0;
function F(){}F.prototype=new N;F.prototype.constructor=F;F.prototype.d=function(){return this};F.prototype.$classData=L({T:0},"scala.scalajs.runtime.RuntimeString$",{T:1,a:1});var ha=void 0;function W(){}W.prototype=new N;W.prototype.constructor=W;function Ba(){}Ba.prototype=W.prototype;var ea=L({U:0},"scala.runtime.BoxedUnit",{U:1,a:1,c:1},void 0,void 0,function(a){return void 0===a}),da=L({J:0},"java.lang.Boolean",{J:1,a:1,c:1,e:1},void 0,void 0,function(a){return"boolean"===typeof a});
function X(){this.v=null}X.prototype=new N;X.prototype.constructor=X;X.prototype.d=function(){Y=this;this.v=I(0,0);return this};
function Ca(a,c){if(0===(-2097152&c))c=""+(4294967296*c+ +(a>>>0));else{var b=(32+q(1E9)|0)-(0!==c?q(c):32+q(a)|0)|0,d=b,f=0===(32&d)?1E9<<d:0,d=0===(32&d)?5E8>>>(31-d|0)|0|0<<d:1E9<<d,g=a,l=c;for(a=c=0;0<=b&&0!==(-2097152&l);){var h=g,t=l,za=f,ia=d;if(t===ia?(-2147483648^h)>=(-2147483648^za):(-2147483648^t)>=(-2147483648^ia))h=l,t=d,l=g-f|0,h=(-2147483648^l)>(-2147483648^g)?-1+(h-t|0)|0:h-t|0,g=l,l=h,32>b?c|=1<<b:a|=1<<b;b=-1+b|0;h=d>>>1|0;f=f>>>1|0|d<<31;d=h}b=l;if(0===b?-1147483648<=(-2147483648^
g):-2147483648<=(-2147483648^b))b=4294967296*l+ +(g>>>0),g=b/1E9,f=g/4294967296|0,d=c,c=g=d+(g|0)|0,a=(-2147483648^g)<(-2147483648^d)?1+(a+f|0)|0:a+f|0,g=b%1E9|0;b=""+g;c=""+(4294967296*a+ +(c>>>0))+"000000000".substring(b.length|0)+b}return c}X.prototype.$classData=L({S:0},"scala.scalajs.runtime.RuntimeLong$",{S:1,a:1,Y:1,c:1});var Y=void 0;function M(){Y||(Y=(new X).d());return Y}
var y=L({I:0},"java.lang.String",{I:1,a:1,c:1,W:1,e:1},void 0,void 0,function(a){return"string"===typeof a}),A=L({K:0},"java.lang.Byte",{K:1,i:1,a:1,c:1,e:1},void 0,void 0,function(a){return z(a)}),ca=L({M:0},"java.lang.Double",{M:1,i:1,a:1,c:1,e:1},void 0,void 0,function(a){return"number"===typeof a}),E=L({N:0},"java.lang.Float",{N:1,i:1,a:1,c:1,e:1},void 0,void 0,function(a){return"number"===typeof a}),D=L({O:0},"java.lang.Integer",{O:1,i:1,a:1,c:1,e:1},void 0,void 0,function(a){return"number"===
typeof a&&(a|0)===a&&1/a!==1/-0}),fa=L({P:0},"java.lang.Long",{P:1,i:1,a:1,c:1,e:1},void 0,void 0,function(a){return!!(a&&a.$classData&&a.$classData.f.z)}),C=L({Q:0},"java.lang.Short",{Q:1,i:1,a:1,c:1,e:1},void 0,void 0,function(a){return B(a)});function Z(){this.r=this.t=0}Z.prototype=new Ba;Z.prototype.constructor=Z;Z.prototype.A=function(){M();var a=this.t,c=this.r;return c===a>>31?""+a:0>c?"-"+Ca(-a|0,0!==a?~c:-c|0):Ca(a,c)};function I(a,c){var b=new Z;b.t=a;b.r=c;return b}
Z.prototype.y=function(){return this.t^this.r};Z.prototype.$classData=L({z:0},"scala.scalajs.runtime.RuntimeLong",{z:1,i:1,a:1,c:1,e:1});m.addClickedMessage=function(){Aa()};var Da=U(),Ea;Ea=new (pa(y).x)([]);(function(a){(0,S().h)(function(){return function(){ya()}}(a))})(Da,Ea);
}).call(this);
//# sourceMappingURL=visscala-opt.js.map
