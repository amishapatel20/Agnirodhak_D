var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __decorateClass = (decorators, target, key, kind) => {
  var result = kind > 1 ? void 0 : kind ? __getOwnPropDesc(target, key) : target;
  for (var i4 = decorators.length - 1, decorator; i4 >= 0; i4--)
    if (decorator = decorators[i4])
      result = (kind ? decorator(target, key, result) : decorator(result)) || result;
  if (kind && result) __defProp(target, key, result);
  return result;
};

// node_modules/@lit/reactive-element/css-tag.js
var t = globalThis;
var e = t.ShadowRoot && (void 0 === t.ShadyCSS || t.ShadyCSS.nativeShadow) && "adoptedStyleSheets" in Document.prototype && "replace" in CSSStyleSheet.prototype;
var s = Symbol();
var o = /* @__PURE__ */ new WeakMap();
var n = class {
  constructor(t3, e7, o5) {
    if (this._$cssResult$ = true, o5 !== s) throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");
    this.cssText = t3, this.t = e7;
  }
  get styleSheet() {
    let t3 = this.o;
    const s2 = this.t;
    if (e && void 0 === t3) {
      const e7 = void 0 !== s2 && 1 === s2.length;
      e7 && (t3 = o.get(s2)), void 0 === t3 && ((this.o = t3 = new CSSStyleSheet()).replaceSync(this.cssText), e7 && o.set(s2, t3));
    }
    return t3;
  }
  toString() {
    return this.cssText;
  }
};
var r = (t3) => new n("string" == typeof t3 ? t3 : t3 + "", void 0, s);
var i = (t3, ...e7) => {
  const o5 = 1 === t3.length ? t3[0] : e7.reduce((e8, s2, o6) => e8 + ((t4) => {
    if (true === t4._$cssResult$) return t4.cssText;
    if ("number" == typeof t4) return t4;
    throw Error("Value passed to 'css' function must be a 'css' function result: " + t4 + ". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.");
  })(s2) + t3[o6 + 1], t3[0]);
  return new n(o5, t3, s);
};
var S = (s2, o5) => {
  if (e) s2.adoptedStyleSheets = o5.map((t3) => t3 instanceof CSSStyleSheet ? t3 : t3.styleSheet);
  else for (const e7 of o5) {
    const o6 = document.createElement("style"), n5 = t.litNonce;
    void 0 !== n5 && o6.setAttribute("nonce", n5), o6.textContent = e7.cssText, s2.appendChild(o6);
  }
};
var c = e ? (t3) => t3 : (t3) => t3 instanceof CSSStyleSheet ? ((t4) => {
  let e7 = "";
  for (const s2 of t4.cssRules) e7 += s2.cssText;
  return r(e7);
})(t3) : t3;

// node_modules/@lit/reactive-element/reactive-element.js
var { is: i2, defineProperty: e2, getOwnPropertyDescriptor: r2, getOwnPropertyNames: h, getOwnPropertySymbols: o2, getPrototypeOf: n2 } = Object;
var a = globalThis;
var c2 = a.trustedTypes;
var l = c2 ? c2.emptyScript : "";
var p = a.reactiveElementPolyfillSupport;
var d = (t3, s2) => t3;
var u = { toAttribute(t3, s2) {
  switch (s2) {
    case Boolean:
      t3 = t3 ? l : null;
      break;
    case Object:
    case Array:
      t3 = null == t3 ? t3 : JSON.stringify(t3);
  }
  return t3;
}, fromAttribute(t3, s2) {
  let i4 = t3;
  switch (s2) {
    case Boolean:
      i4 = null !== t3;
      break;
    case Number:
      i4 = null === t3 ? null : Number(t3);
      break;
    case Object:
    case Array:
      try {
        i4 = JSON.parse(t3);
      } catch (t4) {
        i4 = null;
      }
  }
  return i4;
} };
var f = (t3, s2) => !i2(t3, s2);
var y = { attribute: true, type: String, converter: u, reflect: false, hasChanged: f };
Symbol.metadata ??= Symbol("metadata"), a.litPropertyMetadata ??= /* @__PURE__ */ new WeakMap();
var b = class extends HTMLElement {
  static addInitializer(t3) {
    this._$Ei(), (this.l ??= []).push(t3);
  }
  static get observedAttributes() {
    return this.finalize(), this._$Eh && [...this._$Eh.keys()];
  }
  static createProperty(t3, s2 = y) {
    if (s2.state && (s2.attribute = false), this._$Ei(), this.elementProperties.set(t3, s2), !s2.noAccessor) {
      const i4 = Symbol(), r5 = this.getPropertyDescriptor(t3, i4, s2);
      void 0 !== r5 && e2(this.prototype, t3, r5);
    }
  }
  static getPropertyDescriptor(t3, s2, i4) {
    const { get: e7, set: h4 } = r2(this.prototype, t3) ?? { get() {
      return this[s2];
    }, set(t4) {
      this[s2] = t4;
    } };
    return { get() {
      return e7?.call(this);
    }, set(s3) {
      const r5 = e7?.call(this);
      h4.call(this, s3), this.requestUpdate(t3, r5, i4);
    }, configurable: true, enumerable: true };
  }
  static getPropertyOptions(t3) {
    return this.elementProperties.get(t3) ?? y;
  }
  static _$Ei() {
    if (this.hasOwnProperty(d("elementProperties"))) return;
    const t3 = n2(this);
    t3.finalize(), void 0 !== t3.l && (this.l = [...t3.l]), this.elementProperties = new Map(t3.elementProperties);
  }
  static finalize() {
    if (this.hasOwnProperty(d("finalized"))) return;
    if (this.finalized = true, this._$Ei(), this.hasOwnProperty(d("properties"))) {
      const t4 = this.properties, s2 = [...h(t4), ...o2(t4)];
      for (const i4 of s2) this.createProperty(i4, t4[i4]);
    }
    const t3 = this[Symbol.metadata];
    if (null !== t3) {
      const s2 = litPropertyMetadata.get(t3);
      if (void 0 !== s2) for (const [t4, i4] of s2) this.elementProperties.set(t4, i4);
    }
    this._$Eh = /* @__PURE__ */ new Map();
    for (const [t4, s2] of this.elementProperties) {
      const i4 = this._$Eu(t4, s2);
      void 0 !== i4 && this._$Eh.set(i4, t4);
    }
    this.elementStyles = this.finalizeStyles(this.styles);
  }
  static finalizeStyles(s2) {
    const i4 = [];
    if (Array.isArray(s2)) {
      const e7 = new Set(s2.flat(1 / 0).reverse());
      for (const s3 of e7) i4.unshift(c(s3));
    } else void 0 !== s2 && i4.push(c(s2));
    return i4;
  }
  static _$Eu(t3, s2) {
    const i4 = s2.attribute;
    return false === i4 ? void 0 : "string" == typeof i4 ? i4 : "string" == typeof t3 ? t3.toLowerCase() : void 0;
  }
  constructor() {
    super(), this._$Ep = void 0, this.isUpdatePending = false, this.hasUpdated = false, this._$Em = null, this._$Ev();
  }
  _$Ev() {
    this._$ES = new Promise((t3) => this.enableUpdating = t3), this._$AL = /* @__PURE__ */ new Map(), this._$E_(), this.requestUpdate(), this.constructor.l?.forEach((t3) => t3(this));
  }
  addController(t3) {
    (this._$EO ??= /* @__PURE__ */ new Set()).add(t3), void 0 !== this.renderRoot && this.isConnected && t3.hostConnected?.();
  }
  removeController(t3) {
    this._$EO?.delete(t3);
  }
  _$E_() {
    const t3 = /* @__PURE__ */ new Map(), s2 = this.constructor.elementProperties;
    for (const i4 of s2.keys()) this.hasOwnProperty(i4) && (t3.set(i4, this[i4]), delete this[i4]);
    t3.size > 0 && (this._$Ep = t3);
  }
  createRenderRoot() {
    const t3 = this.shadowRoot ?? this.attachShadow(this.constructor.shadowRootOptions);
    return S(t3, this.constructor.elementStyles), t3;
  }
  connectedCallback() {
    this.renderRoot ??= this.createRenderRoot(), this.enableUpdating(true), this._$EO?.forEach((t3) => t3.hostConnected?.());
  }
  enableUpdating(t3) {
  }
  disconnectedCallback() {
    this._$EO?.forEach((t3) => t3.hostDisconnected?.());
  }
  attributeChangedCallback(t3, s2, i4) {
    this._$AK(t3, i4);
  }
  _$EC(t3, s2) {
    const i4 = this.constructor.elementProperties.get(t3), e7 = this.constructor._$Eu(t3, i4);
    if (void 0 !== e7 && true === i4.reflect) {
      const r5 = (void 0 !== i4.converter?.toAttribute ? i4.converter : u).toAttribute(s2, i4.type);
      this._$Em = t3, null == r5 ? this.removeAttribute(e7) : this.setAttribute(e7, r5), this._$Em = null;
    }
  }
  _$AK(t3, s2) {
    const i4 = this.constructor, e7 = i4._$Eh.get(t3);
    if (void 0 !== e7 && this._$Em !== e7) {
      const t4 = i4.getPropertyOptions(e7), r5 = "function" == typeof t4.converter ? { fromAttribute: t4.converter } : void 0 !== t4.converter?.fromAttribute ? t4.converter : u;
      this._$Em = e7, this[e7] = r5.fromAttribute(s2, t4.type), this._$Em = null;
    }
  }
  requestUpdate(t3, s2, i4) {
    if (void 0 !== t3) {
      if (i4 ??= this.constructor.getPropertyOptions(t3), !(i4.hasChanged ?? f)(this[t3], s2)) return;
      this.P(t3, s2, i4);
    }
    false === this.isUpdatePending && (this._$ES = this._$ET());
  }
  P(t3, s2, i4) {
    this._$AL.has(t3) || this._$AL.set(t3, s2), true === i4.reflect && this._$Em !== t3 && (this._$Ej ??= /* @__PURE__ */ new Set()).add(t3);
  }
  async _$ET() {
    this.isUpdatePending = true;
    try {
      await this._$ES;
    } catch (t4) {
      Promise.reject(t4);
    }
    const t3 = this.scheduleUpdate();
    return null != t3 && await t3, !this.isUpdatePending;
  }
  scheduleUpdate() {
    return this.performUpdate();
  }
  performUpdate() {
    if (!this.isUpdatePending) return;
    if (!this.hasUpdated) {
      if (this.renderRoot ??= this.createRenderRoot(), this._$Ep) {
        for (const [t5, s3] of this._$Ep) this[t5] = s3;
        this._$Ep = void 0;
      }
      const t4 = this.constructor.elementProperties;
      if (t4.size > 0) for (const [s3, i4] of t4) true !== i4.wrapped || this._$AL.has(s3) || void 0 === this[s3] || this.P(s3, this[s3], i4);
    }
    let t3 = false;
    const s2 = this._$AL;
    try {
      t3 = this.shouldUpdate(s2), t3 ? (this.willUpdate(s2), this._$EO?.forEach((t4) => t4.hostUpdate?.()), this.update(s2)) : this._$EU();
    } catch (s3) {
      throw t3 = false, this._$EU(), s3;
    }
    t3 && this._$AE(s2);
  }
  willUpdate(t3) {
  }
  _$AE(t3) {
    this._$EO?.forEach((t4) => t4.hostUpdated?.()), this.hasUpdated || (this.hasUpdated = true, this.firstUpdated(t3)), this.updated(t3);
  }
  _$EU() {
    this._$AL = /* @__PURE__ */ new Map(), this.isUpdatePending = false;
  }
  get updateComplete() {
    return this.getUpdateComplete();
  }
  getUpdateComplete() {
    return this._$ES;
  }
  shouldUpdate(t3) {
    return true;
  }
  update(t3) {
    this._$Ej &&= this._$Ej.forEach((t4) => this._$EC(t4, this[t4])), this._$EU();
  }
  updated(t3) {
  }
  firstUpdated(t3) {
  }
};
b.elementStyles = [], b.shadowRootOptions = { mode: "open" }, b[d("elementProperties")] = /* @__PURE__ */ new Map(), b[d("finalized")] = /* @__PURE__ */ new Map(), p?.({ ReactiveElement: b }), (a.reactiveElementVersions ??= []).push("2.0.4");

// node_modules/lit-html/lit-html.js
var n3 = globalThis;
var c3 = n3.trustedTypes;
var h2 = c3 ? c3.createPolicy("lit-html", { createHTML: (t3) => t3 }) : void 0;
var f2 = "$lit$";
var v = `lit$${Math.random().toFixed(9).slice(2)}$`;
var m = "?" + v;
var _ = `<${m}>`;
var w = document;
var lt = () => w.createComment("");
var st = (t3) => null === t3 || "object" != typeof t3 && "function" != typeof t3;
var g = Array.isArray;
var $ = (t3) => g(t3) || "function" == typeof t3?.[Symbol.iterator];
var x = "[ 	\n\f\r]";
var T = /<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g;
var E = /-->/g;
var k = />/g;
var O = RegExp(`>|${x}(?:([^\\s"'>=/]+)(${x}*=${x}*(?:[^ 	
\f\r"'\`<>=]|("|')|))|$)`, "g");
var S2 = /'/g;
var j = /"/g;
var M = /^(?:script|style|textarea|title)$/i;
var P = (t3) => (i4, ...s2) => ({ _$litType$: t3, strings: i4, values: s2 });
var ke = P(1);
var Oe = P(2);
var Se = P(3);
var R = Symbol.for("lit-noChange");
var D = Symbol.for("lit-nothing");
var V = /* @__PURE__ */ new WeakMap();
var I = w.createTreeWalker(w, 129);
function N(t3, i4) {
  if (!g(t3) || !t3.hasOwnProperty("raw")) throw Error("invalid template strings array");
  return void 0 !== h2 ? h2.createHTML(i4) : i4;
}
var U = (t3, i4) => {
  const s2 = t3.length - 1, e7 = [];
  let h4, o5 = 2 === i4 ? "<svg>" : 3 === i4 ? "<math>" : "", n5 = T;
  for (let i5 = 0; i5 < s2; i5++) {
    const s3 = t3[i5];
    let r5, l2, c4 = -1, a2 = 0;
    for (; a2 < s3.length && (n5.lastIndex = a2, l2 = n5.exec(s3), null !== l2); ) a2 = n5.lastIndex, n5 === T ? "!--" === l2[1] ? n5 = E : void 0 !== l2[1] ? n5 = k : void 0 !== l2[2] ? (M.test(l2[2]) && (h4 = RegExp("</" + l2[2], "g")), n5 = O) : void 0 !== l2[3] && (n5 = O) : n5 === O ? ">" === l2[0] ? (n5 = h4 ?? T, c4 = -1) : void 0 === l2[1] ? c4 = -2 : (c4 = n5.lastIndex - l2[2].length, r5 = l2[1], n5 = void 0 === l2[3] ? O : '"' === l2[3] ? j : S2) : n5 === j || n5 === S2 ? n5 = O : n5 === E || n5 === k ? n5 = T : (n5 = O, h4 = void 0);
    const u2 = n5 === O && t3[i5 + 1].startsWith("/>") ? " " : "";
    o5 += n5 === T ? s3 + _ : c4 >= 0 ? (e7.push(r5), s3.slice(0, c4) + f2 + s3.slice(c4) + v + u2) : s3 + v + (-2 === c4 ? i5 : u2);
  }
  return [N(t3, o5 + (t3[s2] || "<?>") + (2 === i4 ? "</svg>" : 3 === i4 ? "</math>" : "")), e7];
};
var B = class _B {
  constructor({ strings: t3, _$litType$: i4 }, s2) {
    let e7;
    this.parts = [];
    let h4 = 0, o5 = 0;
    const n5 = t3.length - 1, r5 = this.parts, [l2, a2] = U(t3, i4);
    if (this.el = _B.createElement(l2, s2), I.currentNode = this.el.content, 2 === i4 || 3 === i4) {
      const t4 = this.el.content.firstChild;
      t4.replaceWith(...t4.childNodes);
    }
    for (; null !== (e7 = I.nextNode()) && r5.length < n5; ) {
      if (1 === e7.nodeType) {
        if (e7.hasAttributes()) for (const t4 of e7.getAttributeNames()) if (t4.endsWith(f2)) {
          const i5 = a2[o5++], s3 = e7.getAttribute(t4).split(v), n6 = /([.?@])?(.*)/.exec(i5);
          r5.push({ type: 1, index: h4, name: n6[2], strings: s3, ctor: "." === n6[1] ? Y : "?" === n6[1] ? Z : "@" === n6[1] ? q : G }), e7.removeAttribute(t4);
        } else t4.startsWith(v) && (r5.push({ type: 6, index: h4 }), e7.removeAttribute(t4));
        if (M.test(e7.tagName)) {
          const t4 = e7.textContent.split(v), i5 = t4.length - 1;
          if (i5 > 0) {
            e7.textContent = c3 ? c3.emptyScript : "";
            for (let s3 = 0; s3 < i5; s3++) e7.append(t4[s3], lt()), I.nextNode(), r5.push({ type: 2, index: ++h4 });
            e7.append(t4[i5], lt());
          }
        }
      } else if (8 === e7.nodeType) if (e7.data === m) r5.push({ type: 2, index: h4 });
      else {
        let t4 = -1;
        for (; -1 !== (t4 = e7.data.indexOf(v, t4 + 1)); ) r5.push({ type: 7, index: h4 }), t4 += v.length - 1;
      }
      h4++;
    }
  }
  static createElement(t3, i4) {
    const s2 = w.createElement("template");
    return s2.innerHTML = t3, s2;
  }
};
function z(t3, i4, s2 = t3, e7) {
  if (i4 === R) return i4;
  let h4 = void 0 !== e7 ? s2.o?.[e7] : s2.l;
  const o5 = st(i4) ? void 0 : i4._$litDirective$;
  return h4?.constructor !== o5 && (h4?._$AO?.(false), void 0 === o5 ? h4 = void 0 : (h4 = new o5(t3), h4._$AT(t3, s2, e7)), void 0 !== e7 ? (s2.o ??= [])[e7] = h4 : s2.l = h4), void 0 !== h4 && (i4 = z(t3, h4._$AS(t3, i4.values), h4, e7)), i4;
}
var F = class {
  constructor(t3, i4) {
    this._$AV = [], this._$AN = void 0, this._$AD = t3, this._$AM = i4;
  }
  get parentNode() {
    return this._$AM.parentNode;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  u(t3) {
    const { el: { content: i4 }, parts: s2 } = this._$AD, e7 = (t3?.creationScope ?? w).importNode(i4, true);
    I.currentNode = e7;
    let h4 = I.nextNode(), o5 = 0, n5 = 0, r5 = s2[0];
    for (; void 0 !== r5; ) {
      if (o5 === r5.index) {
        let i5;
        2 === r5.type ? i5 = new et(h4, h4.nextSibling, this, t3) : 1 === r5.type ? i5 = new r5.ctor(h4, r5.name, r5.strings, this, t3) : 6 === r5.type && (i5 = new K(h4, this, t3)), this._$AV.push(i5), r5 = s2[++n5];
      }
      o5 !== r5?.index && (h4 = I.nextNode(), o5++);
    }
    return I.currentNode = w, e7;
  }
  p(t3) {
    let i4 = 0;
    for (const s2 of this._$AV) void 0 !== s2 && (void 0 !== s2.strings ? (s2._$AI(t3, s2, i4), i4 += s2.strings.length - 2) : s2._$AI(t3[i4])), i4++;
  }
};
var et = class _et {
  get _$AU() {
    return this._$AM?._$AU ?? this.v;
  }
  constructor(t3, i4, s2, e7) {
    this.type = 2, this._$AH = D, this._$AN = void 0, this._$AA = t3, this._$AB = i4, this._$AM = s2, this.options = e7, this.v = e7?.isConnected ?? true;
  }
  get parentNode() {
    let t3 = this._$AA.parentNode;
    const i4 = this._$AM;
    return void 0 !== i4 && 11 === t3?.nodeType && (t3 = i4.parentNode), t3;
  }
  get startNode() {
    return this._$AA;
  }
  get endNode() {
    return this._$AB;
  }
  _$AI(t3, i4 = this) {
    t3 = z(this, t3, i4), st(t3) ? t3 === D || null == t3 || "" === t3 ? (this._$AH !== D && this._$AR(), this._$AH = D) : t3 !== this._$AH && t3 !== R && this._(t3) : void 0 !== t3._$litType$ ? this.$(t3) : void 0 !== t3.nodeType ? this.T(t3) : $(t3) ? this.k(t3) : this._(t3);
  }
  O(t3) {
    return this._$AA.parentNode.insertBefore(t3, this._$AB);
  }
  T(t3) {
    this._$AH !== t3 && (this._$AR(), this._$AH = this.O(t3));
  }
  _(t3) {
    this._$AH !== D && st(this._$AH) ? this._$AA.nextSibling.data = t3 : this.T(w.createTextNode(t3)), this._$AH = t3;
  }
  $(t3) {
    const { values: i4, _$litType$: s2 } = t3, e7 = "number" == typeof s2 ? this._$AC(t3) : (void 0 === s2.el && (s2.el = B.createElement(N(s2.h, s2.h[0]), this.options)), s2);
    if (this._$AH?._$AD === e7) this._$AH.p(i4);
    else {
      const t4 = new F(e7, this), s3 = t4.u(this.options);
      t4.p(i4), this.T(s3), this._$AH = t4;
    }
  }
  _$AC(t3) {
    let i4 = V.get(t3.strings);
    return void 0 === i4 && V.set(t3.strings, i4 = new B(t3)), i4;
  }
  k(t3) {
    g(this._$AH) || (this._$AH = [], this._$AR());
    const i4 = this._$AH;
    let s2, e7 = 0;
    for (const h4 of t3) e7 === i4.length ? i4.push(s2 = new _et(this.O(lt()), this.O(lt()), this, this.options)) : s2 = i4[e7], s2._$AI(h4), e7++;
    e7 < i4.length && (this._$AR(s2 && s2._$AB.nextSibling, e7), i4.length = e7);
  }
  _$AR(t3 = this._$AA.nextSibling, i4) {
    for (this._$AP?.(false, true, i4); t3 && t3 !== this._$AB; ) {
      const i5 = t3.nextSibling;
      t3.remove(), t3 = i5;
    }
  }
  setConnected(t3) {
    void 0 === this._$AM && (this.v = t3, this._$AP?.(t3));
  }
};
var G = class {
  get tagName() {
    return this.element.tagName;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  constructor(t3, i4, s2, e7, h4) {
    this.type = 1, this._$AH = D, this._$AN = void 0, this.element = t3, this.name = i4, this._$AM = e7, this.options = h4, s2.length > 2 || "" !== s2[0] || "" !== s2[1] ? (this._$AH = Array(s2.length - 1).fill(new String()), this.strings = s2) : this._$AH = D;
  }
  _$AI(t3, i4 = this, s2, e7) {
    const h4 = this.strings;
    let o5 = false;
    if (void 0 === h4) t3 = z(this, t3, i4, 0), o5 = !st(t3) || t3 !== this._$AH && t3 !== R, o5 && (this._$AH = t3);
    else {
      const e8 = t3;
      let n5, r5;
      for (t3 = h4[0], n5 = 0; n5 < h4.length - 1; n5++) r5 = z(this, e8[s2 + n5], i4, n5), r5 === R && (r5 = this._$AH[n5]), o5 ||= !st(r5) || r5 !== this._$AH[n5], r5 === D ? t3 = D : t3 !== D && (t3 += (r5 ?? "") + h4[n5 + 1]), this._$AH[n5] = r5;
    }
    o5 && !e7 && this.j(t3);
  }
  j(t3) {
    t3 === D ? this.element.removeAttribute(this.name) : this.element.setAttribute(this.name, t3 ?? "");
  }
};
var Y = class extends G {
  constructor() {
    super(...arguments), this.type = 3;
  }
  j(t3) {
    this.element[this.name] = t3 === D ? void 0 : t3;
  }
};
var Z = class extends G {
  constructor() {
    super(...arguments), this.type = 4;
  }
  j(t3) {
    this.element.toggleAttribute(this.name, !!t3 && t3 !== D);
  }
};
var q = class extends G {
  constructor(t3, i4, s2, e7, h4) {
    super(t3, i4, s2, e7, h4), this.type = 5;
  }
  _$AI(t3, i4 = this) {
    if ((t3 = z(this, t3, i4, 0) ?? D) === R) return;
    const s2 = this._$AH, e7 = t3 === D && s2 !== D || t3.capture !== s2.capture || t3.once !== s2.once || t3.passive !== s2.passive, h4 = t3 !== D && (s2 === D || e7);
    e7 && this.element.removeEventListener(this.name, this, s2), h4 && this.element.addEventListener(this.name, this, t3), this._$AH = t3;
  }
  handleEvent(t3) {
    "function" == typeof this._$AH ? this._$AH.call(this.options?.host ?? this.element, t3) : this._$AH.handleEvent(t3);
  }
};
var K = class {
  constructor(t3, i4, s2) {
    this.element = t3, this.type = 6, this._$AN = void 0, this._$AM = i4, this.options = s2;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  _$AI(t3) {
    z(this, t3);
  }
};
var Re = n3.litHtmlPolyfillSupport;
Re?.(B, et), (n3.litHtmlVersions ??= []).push("3.2.0");
var Q = (t3, i4, s2) => {
  const e7 = s2?.renderBefore ?? i4;
  let h4 = e7._$litPart$;
  if (void 0 === h4) {
    const t4 = s2?.renderBefore ?? null;
    e7._$litPart$ = h4 = new et(i4.insertBefore(lt(), t4), t4, void 0, s2 ?? {});
  }
  return h4._$AI(t3), h4;
};

// node_modules/lit-element/lit-element.js
var h3 = class extends b {
  constructor() {
    super(...arguments), this.renderOptions = { host: this }, this.o = void 0;
  }
  createRenderRoot() {
    const t3 = super.createRenderRoot();
    return this.renderOptions.renderBefore ??= t3.firstChild, t3;
  }
  update(t3) {
    const e7 = this.render();
    this.hasUpdated || (this.renderOptions.isConnected = this.isConnected), super.update(t3), this.o = Q(e7, this.renderRoot, this.renderOptions);
  }
  connectedCallback() {
    super.connectedCallback(), this.o?.setConnected(true);
  }
  disconnectedCallback() {
    super.disconnectedCallback(), this.o?.setConnected(false);
  }
  render() {
    return R;
  }
};
h3._$litElement$ = true, h3["finalized"] = true, globalThis.litElementHydrateSupport?.({ LitElement: h3 });
var f3 = globalThis.litElementPolyfillSupport;
f3?.({ LitElement: h3 });
(globalThis.litElementVersions ??= []).push("4.1.0");

// node_modules/@lit/reactive-element/decorators/property.js
var o3 = { attribute: true, type: String, converter: u, reflect: false, hasChanged: f };
var r3 = (t3 = o3, e7, r5) => {
  const { kind: n5, metadata: i4 } = r5;
  let s2 = globalThis.litPropertyMetadata.get(i4);
  if (void 0 === s2 && globalThis.litPropertyMetadata.set(i4, s2 = /* @__PURE__ */ new Map()), s2.set(r5.name, t3), "accessor" === n5) {
    const { name: o5 } = r5;
    return { set(r6) {
      const n6 = e7.get.call(this);
      e7.set.call(this, r6), this.requestUpdate(o5, n6, t3);
    }, init(e8) {
      return void 0 !== e8 && this.P(o5, void 0, t3), e8;
    } };
  }
  if ("setter" === n5) {
    const { name: o5 } = r5;
    return function(r6) {
      const n6 = this[o5];
      e7.call(this, r6), this.requestUpdate(o5, n6, t3);
    };
  }
  throw Error("Unsupported decorator location: " + n5);
};
function n4(t3) {
  return (e7, o5) => "object" == typeof o5 ? r3(t3, e7, o5) : ((t4, e8, o6) => {
    const r5 = e8.hasOwnProperty(o6);
    return e8.constructor.createProperty(o6, r5 ? { ...t4, wrapped: true } : t4), r5 ? Object.getOwnPropertyDescriptor(e8, o6) : void 0;
  })(t3, e7, o5);
}

// node_modules/@lit/reactive-element/decorators/base.js
var e3 = (e7, t3, c4) => (c4.configurable = true, c4.enumerable = true, Reflect.decorate && "object" != typeof t3 && Object.defineProperty(e7, t3, c4), c4);

// node_modules/@lit/reactive-element/decorators/query.js
function e4(e7, r5) {
  return (n5, s2, i4) => {
    const o5 = (t3) => t3.renderRoot?.querySelector(e7) ?? null;
    if (r5) {
      const { get: e8, set: r6 } = "object" == typeof s2 ? n5 : i4 ?? (() => {
        const t3 = Symbol();
        return { get() {
          return this[t3];
        }, set(e9) {
          this[t3] = e9;
        } };
      })();
      return e3(n5, s2, { get() {
        let t3 = e8.call(this);
        return void 0 === t3 && (t3 = o5(this), (null !== t3 || this.hasUpdated) && r6.call(this, t3)), t3;
      } });
    }
    return e3(n5, s2, { get() {
      return o5(this);
    } });
  };
}

// node_modules/@lit/reactive-element/decorators/query-all.js
var e5;
function r4(r5) {
  return (n5, o5) => e3(n5, o5, { get() {
    return (this.renderRoot ?? (e5 ??= document.createDocumentFragment())).querySelectorAll(r5);
  } });
}

// node_modules/@lit/reactive-element/decorators/query-assigned-elements.js
function o4(o5) {
  return (e7, n5) => {
    const { slot: r5, selector: s2 } = o5 ?? {}, c4 = "slot" + (r5 ? `[name=${r5}]` : ":not([name])");
    return e3(e7, n5, { get() {
      const t3 = this.renderRoot?.querySelector(c4), e8 = t3?.assignedElements(o5) ?? [];
      return void 0 === s2 ? e8 : e8.filter((t4) => t4.matches(s2));
    } });
  };
}

// js/ipywidgets_styles.ts
var legacyStyles = i`
    .legacy-button {
        align-items: center;
        background-color: var(--jp-layout-color2);
        border-width: 0;
        box-shadow: none;
        color: var(--jp-ui-font-color1);
        cursor: pointer;
        display: flex;
        font-family: "Helvetica Neue", Arial, Helvetica, sans-serif;
        font-size: var(--jp-widgets-font-size);
        justify-content: center;
        line-height: var(--jp-widgets-inline-height);
        padding: 0;
        user-select: none;
    }

    .legacy-button.primary {
        background-color: var(--jp-brand-color1);
        color: var(--jp-ui-inverse-font-color1);
    }

    .legacy-button:hover:enabled,
    .legacy-button:focus:enabled {
        box-shadow: 0 2px 2px 0
                rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
            0 3px 1px -2px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity)),
            0 1px 5px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity));
    }

    .legacy-slider {
        -webkit-appearance: none;
        appearance: none;
        background: var(--jp-layout-color3);
        border-radius: 3px;
        height: 4px;
        outline: none;
    }

    .legacy-slider::-webkit-slider-thumb,
    .legacy-slider::-moz-range-thumb {
        -moz-appearance: none;
        -webkit-appearance: none;
        appearance: none;
        border-radius: 50%;
        cursor: pointer;
        height: var(--jp-widgets-slider-handle-size);
        width: var(--jp-widgets-slider-handle-size);
    }

    .legacy-text {
        color: var(--jp-widgets-label-color);
        font-family: "Helvetica Neue", Arial, Helvetica, sans-serif;
        font-size: var(--jp-widgets-font-size);
        height: var(--jp-widgets-inline-height);
        line-height: var(--jp-widgets-inline-height);
        vertical-align: middle;
    }

    .legacy-select {
        -moz-appearance: none;
        -webkit-appearance: none;
        appearance: none;
        background-color: var(--jp-widgets-input-background-color);
        background-image: var(--jp-widgets-dropdown-arrow);
        background-position: right center;
        background-repeat: no-repeat;
        background-size: 20px;
        border-radius: 0;
        border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
        box-shadow: none;
        box-sizing: border-box;
        color: var(--jp-widgets-input-color);
        flex: 1 1 var(--jp-widgets-inline-width-short);
        font-size: var(--jp-widgets-font-size);
        height: 28px;
        line-height: 28px;
        min-width: 0;
        outline: none !important;
        padding-left: calc(var(--jp-widgets-input-padding)* 2);
        padding-right: 20px;
        vertical-align: top;
    }

    .legacy-select[disabled] {
        opacity: 0.4;
        cursor: not-allowed;
    }

    .legacy-text-input {
        background: var(--jp-widgets-input-background-color);
        border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
        box-sizing: border-box;
        color: var(--jp-widgets-input-color);
        flex-grow: 1;
        flex-shrink: 1;
        font-size: var(--jp-widgets-font-size);
        height: var(--jp-widgets-inline-height);
        line-height: var(--jp-widgets-inline-height);
        min-width: 0;
        outline: none !important;
        padding: var(--jp-widgets-input-padding) calc(var(--jp-widgets-input-padding) * 2);
    }

    .legacy-text-input:disabled {
        opacity: var(--jp-widgets-disabled-opacity);
    }

    .legacy-color {
        align-self: stretch;
        background: var(--jp-widgets-input-background-color);
        border-left: none;
        border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
        box-sizing: border-box;
        color: var(--jp-widgets-input-color);
        flex-grow: 0;
        flex-shrink: 0;
        height: var(--jp-widgets-inline-height);
        outline: none !important;
        padding: 0 2px;
        width: var(--jp-widgets-inline-height);
    }

    .legacy-radio {
        margin: 0;
        vertical-align: middle;
    }

    .legacy-button.active {
        background-color: var(--colab-primary-surface-color, --jp-layout-color3);
        color: var(--jp-ui-font-color1);
        box-shadow: 0 4px 5px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
                    0 1px 10px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity)),
                    0 2px 4px -1px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity));
    }

    .legacy-button.primary {
        background-color: var(--jp-brand-color1);
        color: var(--jp-ui-inverse-font-color1);
    }

    .legacy-button.primary.active {
        background-color: var(--jp-brand-color0);
        color: var(--jp-ui-inverse-font-color0);
    }

    .legacy-button.active {
        background-color: var(--colab-primary-surface-color, --jp-layout-color3);
        color: var(--jp-ui-font-color1);
        box-shadow: 0 4px 5px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
                    0 1px 10px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity)),
                    0 2px 4px -1px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity));
    }

    .legacy-button.primary {
        background-color: var(--jp-brand-color1);
        color: var(--jp-ui-inverse-font-color1);
    }

    .legacy-button.primary.active {
        background-color: var(--jp-brand-color0);
        color: var(--jp-ui-inverse-font-color0);
    }

    .legacy-select {
        -moz-appearance: none;
        -webkit-appearance: none;
        appearance: none;
        background-color: var(--jp-widgets-input-background-color);
        background-image: var(--jp-widgets-dropdown-arrow);
        background-position: right center;
        background-repeat: no-repeat;
        background-size: 20px;
        border-radius: 0;
        border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
        box-shadow: none;
        box-sizing: border-box;
        color: var(--jp-widgets-input-color);
        flex: 1 1 var(--jp-widgets-inline-width-short);
        font-size: var(--jp-widgets-font-size);
        height: inherit;
        min-width: 0;
        outline: none !important;
        padding-left: calc(var(--jp-widgets-input-padding)* 2);
        padding-right: 20px;
        vertical-align: top;
    }

    .legacy-input {
        box-sizing: border-box;
        background-color: var(--colab-primary-surface-color, --jp-widgets-input-background-color);
        border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
        color: var(--jp-widgets-input-color);
        flex-grow: 1;
        flex-shrink: 1;
        font-size: var(--jp-widgets-font-size);
        min-width: 0;
        outline: none !important;
        padding: var(--jp-widgets-input-padding) calc(var(--jp-widgets-input-padding)* 2);
        height: var(--jp-widgets-inline-height);
        line-height: var(--jp-widgets-inline-height);
    }
`;

// js/utils.ts
function loadFonts() {
  if (!document.querySelector(".custom-fonts")) {
    const styleElement = document.createElement("style");
    styleElement.classList.add("custom-fonts");
    styleElement.textContent = '@import "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined";';
    document.body.appendChild(styleElement);
  }
}
function reverseMap(map) {
  const reversedMap = /* @__PURE__ */ new Map();
  for (const [key, value] of map.entries()) {
    if (value != null) {
      reversedMap.set(value, key);
    }
  }
  return reversedMap;
}

// js/lit_widget.ts
var LitWidget = class extends h3 {
  constructor() {
    super(...arguments);
    this._model = void 0;
  }
  onCustomMessage(_msg) {
  }
  viewNameToModelName() {
    return reverseMap(this.modelNameToViewName());
  }
  set model(model) {
    this._model = model;
    for (const [modelKey, widgetKey] of this.modelNameToViewName()) {
      if (widgetKey) {
        this[widgetKey] = model.get(modelKey);
        model.on(`change:${String(modelKey)}`, () => {
          this[widgetKey] = model.get(modelKey);
        });
      }
    }
    model.on("msg:custom", (msg) => {
      this.onCustomMessage?.(msg);
    });
  }
  get model() {
    return this._model;
  }
  updated(changedProperties) {
    const viewToModelMap = this.viewNameToModelName();
    for (const [viewProp, _2] of changedProperties) {
      const castViewProp = viewProp;
      if (viewToModelMap.has(castViewProp)) {
        const modelProp = viewToModelMap.get(castViewProp);
        this._model?.set(
          modelProp,
          this[castViewProp]
        );
      }
    }
    this._model?.save_changes();
  }
};

// js/styles.ts
var materialStyles = i`
    @font-face {
        font-family: "Material Symbols Outlined";
        font-style: normal;
        font-weight: 400;
        src: url(https://fonts.gstatic.com/s/materialsymbolsoutlined/v205/kJF1BvYX7BgnkSrUwT8OhrdQw4oELdPIeeII9v6oDMzByHX9rA6RzaxHMPdY43zj-jCxv3fzvRNU22ZXGJpEpjC_1v-p_4MrImHCIJIZrDCvHOejbd5zrDAt.woff2)
            format("woff2");
    }

    .material-symbols-outlined {
        -webkit-font-feature-settings: "liga";
        -webkit-font-smoothing: antialiased;
        direction: ltr;
        display: inline-block;
        font-family: "Material Symbols Outlined";
        font-style: normal;
        font-weight: normal;
        letter-spacing: normal;
        line-height: 1;
        text-transform: none;
        white-space: nowrap;
        word-wrap: normal;
    }
`;
var flexStyles = i`
    .vertical-flex {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .horizontal-flex {
        align-items: center;
        display: flex;
        flex-wrap: nowrap;
        gap: 8px;
    }

    input:not([type="radio"]):not([type="checkbox"]) {
        width: 100%;
    }

    select {
        width: 100%;
    }

    .horizontal-flex .legacy-text {
        flex-shrink: 0;
    }
`;

// node_modules/lit-html/directive.js
var t2 = { ATTRIBUTE: 1, CHILD: 2, PROPERTY: 3, BOOLEAN_ATTRIBUTE: 4, EVENT: 5, ELEMENT: 6 };
var e6 = (t3) => (...e7) => ({ _$litDirective$: t3, values: e7 });
var i3 = class {
  constructor(t3) {
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  _$AT(t3, e7, i4) {
    this.t = t3, this._$AM = e7, this.i = i4;
  }
  _$AS(t3, e7) {
    return this.update(t3, e7);
  }
  update(t3, e7) {
    return this.render(...e7);
  }
};

// node_modules/lit-html/directives/class-map.js
var Rt = e6(class extends i3 {
  constructor(s2) {
    if (super(s2), s2.type !== t2.ATTRIBUTE || "class" !== s2.name || s2.strings?.length > 2) throw Error("`classMap()` can only be used in the `class` attribute and must be the only part in the attribute.");
  }
  render(t3) {
    return " " + Object.keys(t3).filter((s2) => t3[s2]).join(" ") + " ";
  }
  update(t3, [s2]) {
    if (void 0 === this.st) {
      this.st = /* @__PURE__ */ new Set(), void 0 !== t3.strings && (this.nt = new Set(t3.strings.join(" ").split(/\s/).filter((t4) => "" !== t4)));
      for (const t4 in s2) s2[t4] && !this.nt?.has(t4) && this.st.add(t4);
      return this.render(s2);
    }
    const i4 = t3.element.classList;
    for (const t4 of this.st) t4 in s2 || (i4.remove(t4), this.st.delete(t4));
    for (const t4 in s2) {
      const r5 = !!s2[t4];
      r5 === this.st.has(t4) || this.nt?.has(t4) || (r5 ? (i4.add(t4), this.st.add(t4)) : (i4.remove(t4), this.st.delete(t4)));
    }
    return R;
  }
});

// node_modules/lit-html/directives/style-map.js
var ee = "important";
var ie = " !" + ee;
var se = e6(class extends i3 {
  constructor(e7) {
    if (super(e7), e7.type !== t2.ATTRIBUTE || "style" !== e7.name || e7.strings?.length > 2) throw Error("The `styleMap` directive must be used in the `style` attribute and must be the only part in the attribute.");
  }
  render(t3) {
    return Object.keys(t3).reduce((e7, r5) => {
      const s2 = t3[r5];
      return null == s2 ? e7 : e7 + `${r5 = r5.includes("-") ? r5 : r5.replace(/(?:^(webkit|moz|ms|o)|)(?=[A-Z])/g, "-$&").toLowerCase()}:${s2};`;
    }, "");
  }
  update(t3, [e7]) {
    const { style: r5 } = t3.element;
    if (void 0 === this.ft) return this.ft = new Set(Object.keys(e7)), this.render(e7);
    for (const t4 of this.ft) null == e7[t4] && (this.ft.delete(t4), t4.includes("-") ? r5.removeProperty(t4) : r5[t4] = null);
    for (const t4 in e7) {
      const s2 = e7[t4];
      if (null != s2) {
        this.ft.add(t4);
        const e8 = "string" == typeof s2 && s2.endsWith(ie);
        t4.includes("-") || e8 ? r5.setProperty(t4, e8 ? s2.slice(0, -11) : s2, e8 ? ee : "") : r5[t4] = s2;
      }
    }
    return R;
  }
});

// js/tab_panel.ts
function convertToId(name) {
  return (name || "").trim().replace(" ", "-").toLowerCase();
}
var TabPanel = class extends h3 {
  constructor() {
    super(...arguments);
    this.tabs = [];
    this.index = 0;
    this.mode = 1 /* HIDE_ON_SECOND_CLICK */;
    this.alignment = "left" /* LEFT */;
  }
  static get componentName() {
    return `tab-panel`;
  }
  static {
    this.styles = [
      legacyStyles,
      materialStyles,
      i`
            ::slotted(*) {
                display: none;
            }

            ::slotted(.show-tab) {
                display: block;
            }

            .container {
                padding: 0;
                width: 100%;
            }

            .tab-container {
                align-items: center;
                display: flex;
                flex-direction: row;
            }

            .tab-container.center {
                justify-content: center;
            }

            .tab-container.left {
                justify-content: flex-start;
            }

            .tab-container.right {
                justify-content: flex-end;
            }

            .tab-container button {
                border-radius: 5px;
                height: 28px;
                margin: 2px 0 2px 8px;
                user-select: none;
            }

            .tab-container button.icon {
                font-size: 16px;
                width: 28px;
            }

            .tab-container button.name {
                padding: 0 8px;
            }
        `
    ];
  }
  render() {
    return ke`
            <div class="container">
                <div
                    role="tablist"
                    class="${Rt({
      "tab-container": true,
      "center": this.alignment === "center" /* CENTER */,
      "left": this.alignment === "left" /* LEFT */,
      "right": this.alignment === "right" /* RIGHT */
    })}">
                    ${this.renderTabs()}
                </div>
                <slot @slotchange=${this.updateSlotChildren}></slot>
            </div>
        `;
  }
  update(changedProperties) {
    super.update(changedProperties);
    if (changedProperties.has("index") && changedProperties.get("index") != null) {
      this.updateSlotChildren();
    }
  }
  updateSlotChildren() {
    if (!this.tabContentElements) {
      return;
    }
    this.tabContentElements.forEach((element, i4) => {
      element.classList.remove("show-tab");
      const id = convertToId(this.tabs[i4].name);
      element.setAttribute("id", `tabpanel-${id}-${i4}`);
      element.setAttribute("role", "tabpanel");
      element.setAttribute("aria-labelledby", `tab-${id}-${i4}`);
    });
    this.tabContentElements[this.index]?.classList.add("show-tab");
  }
  renderTabs() {
    return this.tabs.map((tab, i4) => {
      const id = convertToId(this.tabs[i4].name);
      return ke`<button
                            id="tab-${id}-${i4}"
                            class="${Rt({
        "legacy-button": true,
        "active": i4 === this.index,
        "icon": !!tab.icon,
        "name": !!tab.name
      })}"
                            style="${se({
        width: tab.width ? `${tab.width}px` : null
      })}"
                            type="button"
                            role="tab"
                            aria-selected="${i4 === this.index ? true : false}"
                            aria-controls="tabpanel-${id}-${i4}"
                            @click=${() => {
        this.onTabClick(i4);
      }}>
                            ${tab.icon ? ke`<span class="material-symbols-outlined">${tab.icon}</span>` : D}
                            <span>${tab.name}</span>
                        </button>`;
    });
  }
  onTabClick(index) {
    switch (this.mode) {
      case 1 /* HIDE_ON_SECOND_CLICK */:
        this.index = this.index === index ? -1 : index;
        break;
      case 0 /* ALWAYS_SHOW */:
      default:
        this.index = index;
    }
    this.dispatchEvent(new CustomEvent("tab-changed", {
      detail: index
    }));
  }
};
__decorateClass([
  n4({ type: Array })
], TabPanel.prototype, "tabs", 2);
__decorateClass([
  n4({ type: Number })
], TabPanel.prototype, "index", 2);
__decorateClass([
  n4({ type: Number })
], TabPanel.prototype, "mode", 2);
__decorateClass([
  n4({ type: String })
], TabPanel.prototype, "alignment", 2);
__decorateClass([
  r4(".tab")
], TabPanel.prototype, "tabElements", 2);
__decorateClass([
  o4()
], TabPanel.prototype, "tabContentElements", 2);
if (!customElements.get(TabPanel.componentName)) {
  customElements.define(TabPanel.componentName, TabPanel);
}

// node_modules/lit-html/directives/unsafe-html.js
var le = class extends i3 {
  constructor(i4) {
    if (super(i4), this.it = D, i4.type !== t2.CHILD) throw Error(this.constructor.directiveName + "() can only be used in child bindings");
  }
  render(t3) {
    if (t3 === D || null == t3) return this._t = void 0, this.it = t3;
    if (t3 === R) return t3;
    if ("string" != typeof t3) throw Error(this.constructor.directiveName + "() called with a non-string value");
    if (t3 === this.it) return this._t;
    this.it = t3;
    const i4 = [t3];
    return i4.raw = i4, this._t = { _$litType$: this.constructor.resultType, strings: i4, values: [] };
  }
};
le.directiveName = "unsafeHTML", le.resultType = 1;
var ae = e6(le);

// js/container.ts
var Container = class extends LitWidget {
  constructor() {
    super(...arguments);
    this.icon = "";
    this.title = "";
    this.collapsed = false;
    this.hideCloseButton = false;
    this.compactMode = false;
    this.noHeader = false;
    this.reverseHeader = false;
  }
  static get componentName() {
    return `widget-container`;
  }
  static {
    this.styles = [
      legacyStyles,
      materialStyles,
      i`
            .container {
                background: var(--jp-layout-color1);
                border-radius: 4.5px;
                box-shadow: 4px 5px 8px 0px #9e9e9e;
            }

            div {
                background-color: var(--colab-primary-surface-color, --jp-layout-color1, white);
            }

            .header {
                display: flex;
                gap: 4px;
                padding: 4px;
            }

            .reversed {
                flex-direction: row-reverse;
            }

            .icon {
                align-items: center;
                display: flex;
                font-size: 20px;
                height: 28px;
                justify-content: center;
                padding: 0 4px;
            }

            .widget-container {
                padding: 8px 12px 12px 12px;
            }

            .hidden {
                display: none;
            }

            .header-button {
                font-size: 16px;
                height: 28px;
                width: 28px;
            }

            .compact-header-button {
                background: transparent;
                font-size: 16px;
                height: 28px;
                width: 28px;
            }

            .header-text {
                align-content: center;
                flex-grow: 1;
                padding: 0 12px 0 0;
            }

            .left-padding {
                padding-left: 8px;
            }
        `
    ];
  }
  modelNameToViewName() {
    return /* @__PURE__ */ new Map([
      ["icon", "icon"],
      ["collapsed", "collapsed"],
      ["title", "title"],
      ["hide_close_button", "hideCloseButton"]
    ]);
  }
  render() {
    return ke`
            <div class="container">
                ${this.noHeader ? D : this.renderHeader()}
                <div class="widget-container ${this.collapsed ? "hidden" : ""}">
                    <slot></slot>
                </div>
            </div>
        `;
  }
  renderHeader() {
    return this.compactMode ? this.renderCompactHeader() : ke`
            <div class="header ${this.reverseHeader ? "reversed" : ""}">
                ${this.renderIcon()}
                ${this.title ? this.renderTitle() : D}
                ${this.renderCollapseButton()}
                ${this.renderCloseButton()}
            </div>`;
  }
  renderCompactHeader() {
    return ke`<div class="header ${this.reverseHeader ? "reversed" : ""}">
            ${this.renderCollapseButton()}
            ${this.title && !this.collapsed ? this.renderTitle() : D}
            ${this.renderCloseButton()}
        </div>`;
  }
  renderCloseButton() {
    if (this.hideCloseButton) {
      return D;
    }
    return ke`
            <button
                class="legacy-button primary header-button"
                @click="${this.onCloseButtonClicked}"
            >
                <span class="material-symbols-outlined">&#xe5cd;</span>
            </button>
        `;
  }
  renderTitle() {
    return ke`<span
            class="${Rt({
      "legacy-text": true,
      "header-text": true,
      "left-padding": this.compactMode && this.title && !this.reverseHeader
    })}"
            >${this.title}</span
        >`;
  }
  onCloseButtonClicked() {
    this.dispatchEvent(new CustomEvent("close-clicked", {}));
  }
  onCollapseToggled() {
    this.collapsed = !this.collapsed;
    this.dispatchEvent(new CustomEvent("collapse-clicked", {}));
  }
  renderIcon() {
    return ke`<span class="icon material-symbols-outlined">
                        ${this.icon}
                    </span>`;
  }
  renderCollapseButton() {
    let icon;
    if (this.compactMode) {
      icon = this.renderIcon();
    } else if (this.collapsed) {
      icon = ke`<span class="material-symbols-outlined"
                >&#xf830;</span
            >`;
    } else {
      icon = ke`<span class="material-symbols-outlined">&#xf507;</span>`;
    }
    return ke`<button
            class="${Rt({
      "legacy-button": true,
      "header-button": !this.compactMode,
      "compact-header-button": this.compactMode,
      "active": !this.collapsed
    })}"
            class="legacy-button header-button"
            @click="${this.onCollapseToggled}"
        >
            ${icon}
        </button>`;
  }
};
__decorateClass([
  n4({ type: String })
], Container.prototype, "icon", 2);
__decorateClass([
  n4({ type: String })
], Container.prototype, "title", 2);
__decorateClass([
  n4({ type: Boolean })
], Container.prototype, "collapsed", 2);
__decorateClass([
  n4({ type: Boolean })
], Container.prototype, "hideCloseButton", 2);
__decorateClass([
  n4({ type: Boolean })
], Container.prototype, "compactMode", 2);
__decorateClass([
  n4({ type: Boolean })
], Container.prototype, "noHeader", 2);
__decorateClass([
  n4({ type: Boolean })
], Container.prototype, "reverseHeader", 2);
if (!customElements.get(Container.componentName)) {
  customElements.define(Container.componentName, Container);
}

// js/search_bar.ts
var SearchBar = class extends LitWidget {
  constructor() {
    super(...arguments);
    this.collapsed = true;
    this.tab_index = 0;
    this.locationModel = JSON.stringify({
      search: "",
      results: [],
      selected: "",
      additional_html: ""
    });
    this.datasetModel = JSON.stringify({
      search: "",
      results: [],
      selected: "",
      additional_html: ""
    });
  }
  static get componentName() {
    return `search-bar`;
  }
  static {
    this.styles = [
      legacyStyles,
      materialStyles,
      i`
            .row {
                display: flex;
                gap: 6px;
            }

            .input-container {
                max-width: 320px;
            }

            .input-container > p {
                margin: 8px 3px;
            }

            input.search {
                margin: 2px 2px 8px 2px;
                width: calc(100% - 4px);
            }

            ul.results {
                list-style-type: none;
                margin: 0;
                margin-bottom: 4px;
                padding: 8px 0;
            }

            label.result {
                align-items: center;
                display: flex;
                margin-bottom: 4px;
            }

            .import-button, .reset-button {
                margin: 0 2px 2px 2px;
                padding: 0 8px;
                white-space: nowrap;
            }

            .dataset-select {
                margin-bottom: 2px;
                margin-right: 2px;
            }

            .additional-html-container {
                max-height: 300px;
                overflow: auto;
                padding: 8px 0;
            }

            .additional-html-container pre {
                white-space: break-spaces;
            }
        `
    ];
  }
  modelNameToViewName() {
    return /* @__PURE__ */ new Map([
      ["collapsed", "collapsed"],
      ["tab_index", "tab_index"],
      ["location_model", "locationModel"],
      ["dataset_model", "datasetModel"]
    ]);
  }
  render() {
    return ke`
            <widget-container
                icon="search"
                title="Search"
                .collapsed="${this.collapsed}"
                .hideCloseButton="${true}"
                .compactMode="${true}">
                <tab-panel
                    .index="${this.tab_index}"
                    .tabs=${[
      { name: "location", width: 110 },
      { name: "data", width: 110 }
    ]}
                    @tab-changed=${(e7) => {
      this.tab_index = e7.detail;
    }}
                    .mode="${0 /* ALWAYS_SHOW */}">
                    <div class="input-container location-container">
                        ${this.renderLocationSearch()}
                    </div>
                    <div class="input-container dataset-container">
                        ${this.renderDatasetSearch()}
                    </div>
                </tab-panel>
            </widget-container>`;
  }
  renderLocationSearch() {
    const locationModel = JSON.parse(this.locationModel);
    const helpText = ke`<p>
            Find your point of interest (by place name,
            address, or coordinates, e.g. 40,-100)
        </p>`;
    const searchInput = ke`<input
            class="legacy-input search location-search"
            type="search"
            placeholder="Search by location / lat-lon"
            @keydown="${(e7) => {
      if (e7.key === "Enter") {
        e7.preventDefault();
        const locationModel2 = JSON.parse(this.locationModel);
        locationModel2.search = this.locationSearch.value || "";
        this.locationModel = JSON.stringify(locationModel2);
      }
    }}" />`;
    const renderedInputs = [helpText, searchInput];
    if (locationModel.results.length) {
      const results = ke`
                ${locationModel.results.map((result) => ke`
                    <li>
                        <label class="result">
                            <input
                            type="radio"
                            name="location-result"
                            value="${result}"
                            .checked="${locationModel.selected === result}"
                            @input="${(e7) => {
        const input = e7.target;
        const locationModel2 = JSON.parse(this.locationModel);
        locationModel2.selected = input.value || "";
        this.locationModel = JSON.stringify(locationModel2);
      }}" />
                            <span>${result}</span>
                        </label>
                    </li>`)}
            `;
      renderedInputs.push(ke`<ul class="results location-results">
                ${results}
            </ul>`);
    }
    if (locationModel.additional_html) {
      renderedInputs.push(ke`<div class="additional-html-container">
                ${ae(locationModel.additional_html)}
            </div>`);
    }
    if (locationModel.search || locationModel.results.length || locationModel.selected) {
      renderedInputs.push(ke`<button
                    class="legacy-button primary reset-button"
                    @click="${() => {
        this.locationModel = JSON.stringify({
          search: "",
          results: [],
          selected: "",
          additional_html: ""
        });
        if (this.locationSearch) {
          this.locationSearch.value = "";
        }
      }}">Reset</button>`);
    }
    return renderedInputs;
  }
  renderDatasetSearch() {
    const datasetModel = JSON.parse(this.datasetModel);
    const helpText = ke`<p>
            Find a dataset by GEE data catalog name or keywords, e.g. elevation
        </p>`;
    const searchInput = ke`<input
            class="legacy-input search dataset-search"
            type="search"
            placeholder="Search dataset / keywords"
            @keydown="${(e7) => {
      if (e7.key === "Enter") {
        e7.preventDefault();
        const datasetModel2 = JSON.parse(this.datasetModel);
        datasetModel2.search = this.datasetSearch?.value || "";
        this.datasetModel = JSON.stringify(datasetModel2);
      }
    }}" />`;
    const renderedInputs = [helpText, searchInput];
    const importButton = ke`<button
            class="legacy-button primary import-button"
            title="Click to import the selected asset"
            @click="${() => {
      this.model?.send({ type: "click", id: "import" });
    }}">
            Reveal Code
        </button>`;
    const results = ke`
            <select
                class="legacy-select dataset-select"
                @input="${(e7) => {
      const input = e7.target;
      const datasetModel2 = JSON.parse(this.datasetModel);
      datasetModel2.selected = input.value || "";
      this.datasetModel = JSON.stringify(datasetModel2);
    }}">
                ${datasetModel.results.map((result) => ke`
                <option>
                    ${result}
                </option>
                `)}
            </select>
        `;
    renderedInputs.push(
      ke`<div class="row">
                ${importButton}
                ${results}
            </div>`
    );
    if (datasetModel.additional_html) {
      renderedInputs.push(ke`<div class="additional-html-container">
                ${ae(datasetModel.additional_html)}
            </div>`);
    }
    return renderedInputs;
  }
};
__decorateClass([
  n4()
], SearchBar.prototype, "collapsed", 2);
__decorateClass([
  n4()
], SearchBar.prototype, "tab_index", 2);
__decorateClass([
  n4()
], SearchBar.prototype, "locationModel", 2);
__decorateClass([
  n4()
], SearchBar.prototype, "datasetModel", 2);
__decorateClass([
  e4(".location-search")
], SearchBar.prototype, "locationSearch", 2);
__decorateClass([
  r4(".location-results input")
], SearchBar.prototype, "locationResults", 2);
__decorateClass([
  e4(".dataset-search")
], SearchBar.prototype, "datasetSearch", 2);
if (!customElements.get(SearchBar.componentName)) {
  customElements.define(SearchBar.componentName, SearchBar);
}
async function render({ model, el }) {
  loadFonts();
  const row = document.createElement(SearchBar.componentName);
  row.model = model;
  el.appendChild(row);
}
var search_bar_default = { render };
export {
  SearchBar,
  search_bar_default as default
};
/*! Bundled license information:

@lit/reactive-element/css-tag.js:
  (**
   * @license
   * Copyright 2019 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/reactive-element.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/lit-html.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-element/lit-element.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/is-server.js:
  (**
   * @license
   * Copyright 2022 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/custom-element.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/property.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/state.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/event-options.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/base.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-all.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-async.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-assigned-elements.js:
  (**
   * @license
   * Copyright 2021 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

@lit/reactive-element/decorators/query-assigned-nodes.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/directive.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/directives/class-map.js:
  (**
   * @license
   * Copyright 2018 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/directives/style-map.js:
  (**
   * @license
   * Copyright 2018 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)

lit-html/directives/unsafe-html.js:
  (**
   * @license
   * Copyright 2017 Google LLC
   * SPDX-License-Identifier: BSD-3-Clause
   *)
*/
