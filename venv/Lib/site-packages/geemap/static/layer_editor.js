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
  constructor(t3, e7, o4) {
    if (this._$cssResult$ = true, o4 !== s) throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");
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
  const o4 = 1 === t3.length ? t3[0] : e7.reduce((e8, s2, o5) => e8 + ((t4) => {
    if (true === t4._$cssResult$) return t4.cssText;
    if ("number" == typeof t4) return t4;
    throw Error("Value passed to 'css' function must be a 'css' function result: " + t4 + ". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.");
  })(s2) + t3[o5 + 1], t3[0]);
  return new n(o4, t3, s);
};
var S = (s2, o4) => {
  if (e) s2.adoptedStyleSheets = o4.map((t3) => t3 instanceof CSSStyleSheet ? t3 : t3.styleSheet);
  else for (const e7 of o4) {
    const o5 = document.createElement("style"), n5 = t.litNonce;
    void 0 !== n5 && o5.setAttribute("nonce", n5), o5.textContent = e7.cssText, s2.appendChild(o5);
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
  let h4, o4 = 2 === i4 ? "<svg>" : 3 === i4 ? "<math>" : "", n5 = T;
  for (let i5 = 0; i5 < s2; i5++) {
    const s3 = t3[i5];
    let r5, l2, c4 = -1, a2 = 0;
    for (; a2 < s3.length && (n5.lastIndex = a2, l2 = n5.exec(s3), null !== l2); ) a2 = n5.lastIndex, n5 === T ? "!--" === l2[1] ? n5 = E : void 0 !== l2[1] ? n5 = k : void 0 !== l2[2] ? (M.test(l2[2]) && (h4 = RegExp("</" + l2[2], "g")), n5 = O) : void 0 !== l2[3] && (n5 = O) : n5 === O ? ">" === l2[0] ? (n5 = h4 ?? T, c4 = -1) : void 0 === l2[1] ? c4 = -2 : (c4 = n5.lastIndex - l2[2].length, r5 = l2[1], n5 = void 0 === l2[3] ? O : '"' === l2[3] ? j : S2) : n5 === j || n5 === S2 ? n5 = O : n5 === E || n5 === k ? n5 = T : (n5 = O, h4 = void 0);
    const u2 = n5 === O && t3[i5 + 1].startsWith("/>") ? " " : "";
    o4 += n5 === T ? s3 + _ : c4 >= 0 ? (e7.push(r5), s3.slice(0, c4) + f2 + s3.slice(c4) + v + u2) : s3 + v + (-2 === c4 ? i5 : u2);
  }
  return [N(t3, o4 + (t3[s2] || "<?>") + (2 === i4 ? "</svg>" : 3 === i4 ? "</math>" : "")), e7];
};
var B = class _B {
  constructor({ strings: t3, _$litType$: i4 }, s2) {
    let e7;
    this.parts = [];
    let h4 = 0, o4 = 0;
    const n5 = t3.length - 1, r5 = this.parts, [l2, a2] = U(t3, i4);
    if (this.el = _B.createElement(l2, s2), I.currentNode = this.el.content, 2 === i4 || 3 === i4) {
      const t4 = this.el.content.firstChild;
      t4.replaceWith(...t4.childNodes);
    }
    for (; null !== (e7 = I.nextNode()) && r5.length < n5; ) {
      if (1 === e7.nodeType) {
        if (e7.hasAttributes()) for (const t4 of e7.getAttributeNames()) if (t4.endsWith(f2)) {
          const i5 = a2[o4++], s3 = e7.getAttribute(t4).split(v), n6 = /([.?@])?(.*)/.exec(i5);
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
  const o4 = st(i4) ? void 0 : i4._$litDirective$;
  return h4?.constructor !== o4 && (h4?._$AO?.(false), void 0 === o4 ? h4 = void 0 : (h4 = new o4(t3), h4._$AT(t3, s2, e7)), void 0 !== e7 ? (s2.o ??= [])[e7] = h4 : s2.l = h4), void 0 !== h4 && (i4 = z(t3, h4._$AS(t3, i4.values), h4, e7)), i4;
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
    let h4 = I.nextNode(), o4 = 0, n5 = 0, r5 = s2[0];
    for (; void 0 !== r5; ) {
      if (o4 === r5.index) {
        let i5;
        2 === r5.type ? i5 = new et(h4, h4.nextSibling, this, t3) : 1 === r5.type ? i5 = new r5.ctor(h4, r5.name, r5.strings, this, t3) : 6 === r5.type && (i5 = new K(h4, this, t3)), this._$AV.push(i5), r5 = s2[++n5];
      }
      o4 !== r5?.index && (h4 = I.nextNode(), o4++);
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
    let o4 = false;
    if (void 0 === h4) t3 = z(this, t3, i4, 0), o4 = !st(t3) || t3 !== this._$AH && t3 !== R, o4 && (this._$AH = t3);
    else {
      const e8 = t3;
      let n5, r5;
      for (t3 = h4[0], n5 = 0; n5 < h4.length - 1; n5++) r5 = z(this, e8[s2 + n5], i4, n5), r5 === R && (r5 = this._$AH[n5]), o4 ||= !st(r5) || r5 !== this._$AH[n5], r5 === D ? t3 = D : t3 !== D && (t3 += (r5 ?? "") + h4[n5 + 1]), this._$AH[n5] = r5;
    }
    o4 && !e7 && this.j(t3);
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
    const { name: o4 } = r5;
    return { set(r6) {
      const n6 = e7.get.call(this);
      e7.set.call(this, r6), this.requestUpdate(o4, n6, t3);
    }, init(e8) {
      return void 0 !== e8 && this.P(o4, void 0, t3), e8;
    } };
  }
  if ("setter" === n5) {
    const { name: o4 } = r5;
    return function(r6) {
      const n6 = this[o4];
      e7.call(this, r6), this.requestUpdate(o4, n6, t3);
    };
  }
  throw Error("Unsupported decorator location: " + n5);
};
function n4(t3) {
  return (e7, o4) => "object" == typeof o4 ? r3(t3, e7, o4) : ((t4, e8, o5) => {
    const r5 = e8.hasOwnProperty(o5);
    return e8.constructor.createProperty(o5, r5 ? { ...t4, wrapped: true } : t4), r5 ? Object.getOwnPropertyDescriptor(e8, o5) : void 0;
  })(t3, e7, o4);
}

// node_modules/@lit/reactive-element/decorators/base.js
var e3 = (e7, t3, c4) => (c4.configurable = true, c4.enumerable = true, Reflect.decorate && "object" != typeof t3 && Object.defineProperty(e7, t3, c4), c4);

// node_modules/@lit/reactive-element/decorators/query.js
function e4(e7, r5) {
  return (n5, s2, i4) => {
    const o4 = (t3) => t3.renderRoot?.querySelector(e7) ?? null;
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
        return void 0 === t3 && (t3 = o4(this), (null !== t3 || this.hasUpdated) && r6.call(this, t3)), t3;
      } });
    }
    return e3(n5, s2, { get() {
      return o4(this);
    } });
  };
}

// node_modules/@lit/reactive-element/decorators/query-all.js
var e5;
function r4(r5) {
  return (n5, o4) => e3(n5, o4, { get() {
    return (this.renderRoot ?? (e5 ??= document.createDocumentFragment())).querySelectorAll(r5);
  } });
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
async function unpackModels(modelIds, manager) {
  return Promise.all(
    modelIds.map((id) => manager.get_model(id.slice("IPY_MODEL_".length)))
  );
}
async function updateChildren(container, model, property = "children") {
  let children = model.get(property);
  if (!Array.isArray(children)) {
    children = [children];
  }
  const child_models = await unpackModels(children, model.widget_manager);
  const child_views = await Promise.all(
    child_models.map((model2) => model2.widget_manager.create_view(model2))
  );
  container.innerHTML = ``;
  for (const child_view of child_views) {
    container.appendChild(child_view.el);
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
function renderSelect(options, value, callback) {
  const emptyOptions = options.length === 0;
  const newOptions = (emptyOptions ? ["---"] : options).map((option) => {
    const isObject = typeof option === "object";
    const optValue = `${isObject ? option.value : option}`;
    const optLabel = `${isObject ? option.label : option}`;
    return ke`
            <option value="${optValue}" ?selected="${value === optValue}">
                ${optLabel}
            </option>
        `;
  });
  return ke`
        <select
            class="legacy-select"
            @change="${callback}"
            ?disabled=${emptyOptions}
        >
            ${newOptions}
        </select>
    `;
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

// js/color_picker.ts
var ColorPicker = class extends h3 {
  constructor() {
    super(...arguments);
    this.value = "#000000";
  }
  static get componentName() {
    return `color-picker`;
  }
  static {
    this.styles = [
      legacyStyles,
      i`
            .color-swatch {
                border-radius: 0;
                height: 28px;
                padding: 0 2px;
                width: 28px;
            }

            .color-text {
                width: 80px;
            }

            .widget-inline-hbox {
                align-items: baseline;
                box-sizing: border-box;
                display: flex;
                flex-direction: row;
                line-height: 28px;
            }
        `
    ];
  }
  render() {
    return ke`
            <div class="widget-inline-hbox">
                <input
                    type="text"
                    class="legacy-text-input color-text"
                    .value="${this.value}"
                    @change="${this.onValueChanged}"
                >
                <input
                    type="color"
                    class="legacy-color color-swatch"
                    .value="${this.value}"
                    @change="${this.onValueChanged}"
                >
            </div>
        `;
  }
  onValueChanged(event) {
    this.value = event.target.value;
    this.dispatchEvent(new CustomEvent("change", {}));
  }
};
__decorateClass([
  n4({ type: String })
], ColorPicker.prototype, "value", 2);
if (!customElements.get(ColorPicker.componentName)) {
  customElements.define(ColorPicker.componentName, ColorPicker);
}

// js/palette_editor.ts
var PaletteEditor = class extends h3 {
  constructor() {
    super(...arguments);
    this.classesOptions = [
      { label: "Any", value: "any" },
      ...Array.from({ length: 7 }, (_2, i4) => ({
        label: `${i4 + 3}`,
        value: `${i4 + 3}`
      }))
    ];
    this.classes = "any";
    this.colormaps = [];
    this.colormap = "Custom";
    this.palette = "";
    this.paletteTokens = [];
  }
  static get componentName() {
    return `palette-editor`;
  }
  static {
    this.styles = [
      flexStyles,
      legacyStyles,
      materialStyles,
      i`
            .horizontal-flex > .legacy-button {
                flex-shrink: 0;
                height: 28px;
                width: 28px;
            }
        `
    ];
  }
  render() {
    return ke`
            <div class="vertical-flex">
                <div class="horizontal-flex">
                    <span class="legacy-text">Colormap:</span>
                    ${renderSelect(
      this.colormaps,
      this.colormap,
      this.onColormapChanged
    )}
                    <span class="legacy-text">Classes:</span>
                    ${renderSelect(
      this.classesOptions,
      this.classes,
      this.onClassesChanged
    )}
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Palette:</span>
                    <input
                        type="text"
                        class="legacy-text-input"
                        id="palette"
                        name="palette"
                        .value="${this.palette}"
                        ?disabled="${this.colormap !== "Custom"}"
                        @change="${this.onPaletteChanged}"
                    />
                </div>
                <slot></slot>
                <div class="horizontal-flex">
                    <span class="legacy-text">Add/remove color:</span>
                    <color-picker></color-picker>
                    <button
                        class="legacy-button"
                        @click="${this.onAddButtonClicked}"
                    >
                        <span class="material-symbols-outlined">add</span>
                    </button>
                    <button
                        class="legacy-button"
                        @click="${this.onSubtractButtonClicked}"
                    >
                        <span class="material-symbols-outlined">remove</span>
                    </button>
                    <button
                        class="legacy-button"
                        @click="${this.onClearButtonClicked}"
                    >
                        <span class="material-symbols-outlined">ink_eraser</span>
                    </button>
                </div>
            </div>
        `;
  }
  updated(changedProperties) {
    super.updated(changedProperties);
    if (changedProperties.has("palette")) {
      if (this.palette === "") {
        this.paletteTokens = [];
      } else {
        this.paletteTokens = this.palette.split(",").map((color) => color.trim());
      }
    }
  }
  sendOnPaletteChangedEvent() {
    this.dispatchEvent(
      new CustomEvent("calculate-palette", {
        detail: {
          colormap: this.colormap,
          classes: this.classes,
          palette: this.palette
        },
        bubbles: true,
        composed: true
      })
    );
  }
  onClassesChanged(event) {
    const target = event.target;
    this.classes = target.value;
    this.sendOnPaletteChangedEvent();
  }
  onColormapChanged(event) {
    const target = event.target;
    this.colormap = target.value;
    this.sendOnPaletteChangedEvent();
  }
  onPaletteChanged(event) {
    const target = event.target;
    this.palette = target.value;
    this.sendOnPaletteChangedEvent();
  }
  onAddButtonClicked(_event) {
    this.colormap = "Custom";
    this.classes = "any";
    const tokens = [...this.paletteTokens];
    tokens.push(this.colorPicker.value);
    this.palette = tokens.join(", ");
    this.sendOnPaletteChangedEvent();
  }
  onSubtractButtonClicked(_event) {
    this.colormap = "Custom";
    this.classes = "any";
    const tokens = [...this.paletteTokens];
    tokens.pop();
    this.palette = tokens.join(", ");
    this.sendOnPaletteChangedEvent();
  }
  onClearButtonClicked(_event) {
    this.colormap = "Custom";
    this.classes = "any";
    this.palette = "";
    this.sendOnPaletteChangedEvent();
  }
};
__decorateClass([
  n4({ type: Array })
], PaletteEditor.prototype, "classesOptions", 2);
__decorateClass([
  n4({ type: String })
], PaletteEditor.prototype, "classes", 2);
__decorateClass([
  n4({ type: Array })
], PaletteEditor.prototype, "colormaps", 2);
__decorateClass([
  n4({ type: String })
], PaletteEditor.prototype, "colormap", 2);
__decorateClass([
  n4({ type: String })
], PaletteEditor.prototype, "palette", 2);
__decorateClass([
  e4("color-picker")
], PaletteEditor.prototype, "colorPicker", 2);
if (!customElements.get(PaletteEditor.componentName)) {
  customElements.define(PaletteEditor.componentName, PaletteEditor);
}

// js/legend_customization.ts
var LegendCustomization = class extends h3 {
  constructor() {
    super(...arguments);
    this.legendTypes = [
      { label: "Linear", value: "linear" /* Linear */ },
      { label: "Step", value: "step" /* Step */ }
    ];
    this.showLegend = false;
    this.legendType = "linear";
    this.title = "Legend";
    this.labels = [];
  }
  static get componentName() {
    return `legend-customization`;
  }
  static {
    this.styles = [
      flexStyles,
      legacyStyles,
      i`
            .hidden {
                display: none;
            }

            .legend-checkbox {
                vertical-align: middle;
            }
        `
    ];
  }
  render() {
    return ke`
            <div class="vertical-flex">
                <div class="horizontal-flex">
                    <span>
                        <input
                            type="checkbox"
                            class="legend-checkbox"
                            .checked="${this.showLegend}"
                            @change="${this.showLegendToggleChanged}"
                        />
                        <span class="legacy-text">Legend</span>
                    </span>
                </div>
                ${this.renderLegendContents()}
            </div>
        `;
  }
  getLegendData() {
    if (this.showLegend) {
      const data = { type: this.legendType };
      if (this.legendType === "step" /* Step */) {
        data.title = this.title;
        data.labels = this.labels;
      }
      return data;
    }
    return void 0;
  }
  renderLegendTypeRadio(option) {
    return ke`
            <span>
                <input
                    type="radio"
                    class="legacy-radio"
                    id="${option.value}"
                    name="legend-type"
                    value="${option.value}"
                    @click="${this.onLegendTypeChanged}"
                    ?checked="${this.legendType === option.value}"
                />
                <label class="legacy-text">${option.label}</label>
            </span>
        `;
  }
  renderLegendContents() {
    if (this.showLegend) {
      return ke`
                <div class="horizontal-flex">
                    ${this.legendTypes.map(
        (model) => this.renderLegendTypeRadio(model)
      )}
                </div>
                ${this.renderLegendTitleAndLabels()}
            `;
    }
    return D;
  }
  renderLegendTitleAndLabels() {
    if (this.legendType === "step" /* Step */) {
      return ke`
                <div class="horizontal-flex">
                    <span class="legacy-text">Legend title:</span>
                    <input
                        type="text"
                        class="legacy-text-input"
                        id="labels"
                        name="labels"
                        .value="${this.title}"
                        @change="${this.onTitleChanged}"
                    />
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Legend labels:</span>
                    <input
                        type="text"
                        class="legacy-text-input"
                        id="labels"
                        name="labels"
                        .value="${this.labels.join(", ")}"
                        @change="${this.onLabelsChanged}"
                    />
                </div>
            `;
    }
    return D;
  }
  showLegendToggleChanged(event) {
    this.showLegend = event.target.checked;
  }
  onLegendTypeChanged(event) {
    this.legendType = event.target.value;
  }
  onTitleChanged(event) {
    this.title = event.target.value;
  }
  onLabelsChanged(event) {
    const labels = event.target.value;
    this.labels = labels.split(",").map((token) => token.trim());
  }
};
__decorateClass([
  n4({ type: Array })
], LegendCustomization.prototype, "legendTypes", 2);
__decorateClass([
  n4({ type: Boolean })
], LegendCustomization.prototype, "showLegend", 2);
__decorateClass([
  n4({ type: String })
], LegendCustomization.prototype, "legendType", 2);
__decorateClass([
  n4({ type: String })
], LegendCustomization.prototype, "title", 2);
__decorateClass([
  n4({ type: String })
], LegendCustomization.prototype, "labels", 2);
if (!customElements.get(LegendCustomization.componentName)) {
  customElements.define(
    LegendCustomization.componentName,
    LegendCustomization
  );
}

// js/raster_layer_editor.ts
var RasterLayerEditor = class extends h3 {
  constructor() {
    super(...arguments);
    this.stretchOptions = [
      { label: "Custom", value: "custom" },
      { label: "1\u03C3", value: "sigma-1" },
      { label: "2\u03C3", value: "sigma-2" },
      { label: "3\u03C3", value: "sigma-3" },
      { label: "90%", value: "percent-90" },
      { label: "98%", value: "percent-98" },
      { label: "100%", value: "percent-100" }
    ];
    this.colorModels = [
      { label: "1 band (Grayscale)", value: "gray" /* Gray */ },
      { label: "3 bands (RGB)", value: "rgb" /* RGB */ }
    ];
    this.colorRamps = [
      { label: "Gamma", value: "gamma" /* Gamma */ },
      { label: "Palette", value: "palette" /* Palette */ }
    ];
    this.colorModel = "";
    this.bandNames = [];
    this.selectedBands = [];
    this.stretch = this.stretchOptions[0].value;
    this.minValue = 0;
    this.maxValue = 1;
    this.minAndMaxValuesLocked = false;
    this.opacity = 1;
    this.colorRamp = "gamma" /* Gamma */;
    this.gamma = 1;
    this.colormaps = [];
  }
  static get componentName() {
    return `raster-layer-editor`;
  }
  static {
    this.styles = [
      flexStyles,
      legacyStyles,
      materialStyles,
      i`
            .horizontal-flex > button {
                height: 28px;
                width: 28px;
            }
        `
    ];
  }
  connectedCallback() {
    super.connectedCallback();
    this.colorModel = this.bandNames.length > 1 ? "rgb" /* RGB */ : "gray" /* Gray */;
  }
  getVisualizationOptions() {
    const visOptions = {
      bands: this.selectedBands,
      min: this.minValue,
      max: this.maxValue,
      opacity: this.opacity
    };
    if (this.colorModel === "gray" /* Gray */) {
      if (this.colorRamp === "palette" /* Palette */) {
        visOptions.palette = this.paletteEditor?.paletteTokens ?? [];
      } else if (this.colorRamp === "gamma" /* Gamma */) {
        visOptions.gamma = this.gamma;
      }
    } else if (this.colorModel === "rgb" /* RGB */) {
      visOptions.gamma = this.gamma;
    }
    if (this.legendCustomization) {
      visOptions.legend = this.legendCustomization.getLegendData();
    }
    return visOptions;
  }
  render() {
    return ke`
            <div class="vertical-flex">
                <div class="horizontal-flex">
                    ${this.colorModels.map(
      (model) => this.renderColorModelRadio(model)
    )}
                </div>
                <div id="band-selection" class="horizontal-flex">
                    ${this.selectedBands.map(
      (band) => this.renderBandSelection(band)
    )}
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Stretch:</span>
                    ${renderSelect(
      this.stretchOptions,
      this.stretch,
      this.onStretchChanged
    )}
                    <button
                        class="legacy-button"
                        @click="${this.onRefreshButtonClicked}"
                    >
                        <span class="material-symbols-outlined">refresh</span>
                    </button>
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Range:</span>
                    <input
                        type="text"
                        class="legacy-text-input"
                        id="min"
                        name="min"
                        .value="${this.minValue ?? "Loading..."}"
                        @keydown="${this.onInputKeyDown}"
                        @change="${this.onMinTextChanged}"
                        ?disabled="${this.minAndMaxValuesLocked}"
                    />
                    <span class="legacy-text">to</span>
                    <input
                        type="text"
                        class="legacy-text-input"
                        id="max"
                        name="max"
                        .value="${this.maxValue ?? "Loading..."}"
                        @keydown="${this.onInputKeyDown}"
                        @change="${this.onMaxTextChanged}"
                        ?disabled="${this.minAndMaxValuesLocked}"
                    />
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Opacity:</span>
                    <input
                        type="range"
                        class="legacy-slider"
                        id="opacity"
                        name="opacity"
                        min="0"
                        max="1.0"
                        step="0.01"
                        .value=${this.opacity}
                        @input=${this.onOpacityChanged}
                    />
                    <span class="legacy-text">${this.opacity.toFixed(2)}</span>
                </div>
                ${this.renderPaletteGammaSelector()}
                ${this.renderPaletteEditor()}
                ${this.renderGammaSlider()}
                ${this.renderLegendCustomization()}
            </div>
        `;
  }
  renderPaletteGammaSelector() {
    if (this.colorModel === "gray" /* Gray */) {
      return ke`
                <div class="horizontal-flex">
                    ${this.colorRamps.map(
        (model) => this.renderColorRampRadio(model)
      )}
                </div>
            `;
    }
    return D;
  }
  renderPaletteEditor() {
    if (this.colorRamp === "palette" /* Palette */ && this.colorModel === "gray" /* Gray */) {
      return ke`
                <palette-editor .colormaps="${this.colormaps}">
                    <slot></slot>
                </palette-editor>
            `;
    }
    return D;
  }
  renderLegendCustomization() {
    if (this.colorRamp === "palette" /* Palette */ && this.colorModel === "gray" /* Gray */) {
      return ke`<legend-customization></legend-customization>`;
    }
    return D;
  }
  renderGammaSlider() {
    if (this.colorRamp === "gamma" /* Gamma */ || this.colorModel === "rgb" /* RGB */) {
      return ke`
                <div class="horizontal-flex">
                    <span class="legacy-text">Gamma:</span>
                    <input
                        type="range"
                        class="legacy-slider"
                        id="gamma"
                        name="gamma"
                        min="0.1"
                        max="10"
                        step="0.01"
                        .value=${this.gamma}
                        @input=${this.onGammaChanged}
                    />
                    <span class="legacy-text">${this.gamma.toFixed(2)}</span>
                </div>
            `;
    }
    return D;
  }
  renderColorModelRadio(option) {
    return ke`
            <span>
                <input
                    type="radio"
                    class="legacy-radio"
                    id="${option.value}"
                    name="color-model"
                    value="${option.value}"
                    @click="${this.onColorModelChanged}"
                    ?checked="${this.colorModel === option.value}"
                />
                <label class="legacy-text">${option.label}</label>
            </span>
        `;
  }
  renderColorRampRadio(option) {
    return ke`
            <span>
                <input
                    type="radio"
                    class="legacy-radio"
                    id="${option.value}"
                    name="color-ramp"
                    value="${option.value}"
                    @click="${this.onColorRampChanged}"
                    ?checked="${this.colorRamp === option.value}"
                />
                <label class="legacy-text">${option.label}</label>
            </span>
        `;
  }
  onColorRampChanged(event) {
    this.colorRamp = event.target.value;
  }
  renderBandSelection(value) {
    return renderSelect(this.bandNames, value, this.onBandSelectionChanged);
  }
  onRefreshButtonClicked(event) {
    event.stopImmediatePropagation();
    this.calculateBandStats();
  }
  onOpacityChanged(event) {
    this.opacity = event.target.valueAsNumber;
  }
  onGammaChanged(event) {
    this.gamma = event.target.valueAsNumber;
  }
  onInputKeyDown(event) {
    event.stopPropagation();
  }
  onMinTextChanged(_event) {
    this.minValue = +this.minInput.value;
  }
  onMaxTextChanged(_event) {
    this.maxValue = +this.maxInput.value;
  }
  calculateBandStats() {
    if (this.stretch === "custom") {
      return;
    }
    this.minValue = void 0;
    this.maxValue = void 0;
    this.dispatchEvent(
      new CustomEvent("calculate-band-stats", {
        bubbles: true,
        composed: true
      })
    );
  }
  onStretchChanged(event) {
    this.stretch = event.target.value;
    this.minAndMaxValuesLocked = this.stretch !== "custom";
  }
  onBandSelectionChanged(_event) {
    this.selectedBands = this.getSelectedBands();
  }
  onColorModelChanged(event) {
    this.colorModel = event.target.value;
  }
  updated(changedProperties) {
    super.updated(changedProperties);
    if (changedProperties.has("colorModel")) {
      if (this.colorModel === "gray" /* Gray */) {
        this.selectedBands = [this.bandNames[0]];
      } else if (this.colorModel == "rgb" /* RGB */) {
        this.selectedBands = [
          this.bandNames[0],
          this.bandNames[0],
          this.bandNames[0]
        ];
      }
    }
    if (changedProperties.has("selectedBands") || changedProperties.has("stretch")) {
      this.calculateBandStats();
    }
  }
  getSelectedBands() {
    return Array.from(this.bandSelects).map((input) => input.value);
  }
};
__decorateClass([
  n4({ type: Array })
], RasterLayerEditor.prototype, "stretchOptions", 2);
__decorateClass([
  n4({ type: Array })
], RasterLayerEditor.prototype, "colorModels", 2);
__decorateClass([
  n4({ type: Array })
], RasterLayerEditor.prototype, "colorRamps", 2);
__decorateClass([
  n4({ type: String })
], RasterLayerEditor.prototype, "colorModel", 2);
__decorateClass([
  n4({ type: Array })
], RasterLayerEditor.prototype, "bandNames", 2);
__decorateClass([
  n4({ type: Array })
], RasterLayerEditor.prototype, "selectedBands", 2);
__decorateClass([
  n4({ type: String })
], RasterLayerEditor.prototype, "stretch", 2);
__decorateClass([
  n4({ type: Number })
], RasterLayerEditor.prototype, "minValue", 2);
__decorateClass([
  n4({ type: Number })
], RasterLayerEditor.prototype, "maxValue", 2);
__decorateClass([
  n4({ type: Boolean })
], RasterLayerEditor.prototype, "minAndMaxValuesLocked", 2);
__decorateClass([
  n4({ type: Number })
], RasterLayerEditor.prototype, "opacity", 2);
__decorateClass([
  n4({ type: String })
], RasterLayerEditor.prototype, "colorRamp", 2);
__decorateClass([
  n4({ type: Number })
], RasterLayerEditor.prototype, "gamma", 2);
__decorateClass([
  n4({ type: Array })
], RasterLayerEditor.prototype, "colormaps", 2);
__decorateClass([
  e4("palette-editor")
], RasterLayerEditor.prototype, "paletteEditor", 2);
__decorateClass([
  e4("legend-customization")
], RasterLayerEditor.prototype, "legendCustomization", 2);
__decorateClass([
  r4("#band-selection select")
], RasterLayerEditor.prototype, "bandSelects", 2);
__decorateClass([
  e4("#min")
], RasterLayerEditor.prototype, "minInput", 2);
__decorateClass([
  e4("#max")
], RasterLayerEditor.prototype, "maxInput", 2);
if (!customElements.get(RasterLayerEditor.componentName)) {
  customElements.define(RasterLayerEditor.componentName, RasterLayerEditor);
}

// js/vector_layer_editor.ts
var _VectorLayerEditor = class _VectorLayerEditor extends h3 {
  constructor() {
    super(...arguments);
    this.colormaps = [];
    this.newLayerName = "";
    this.opacity = 1;
    this.pointSize = 3;
    this.pointShape = _VectorLayerEditor.pointShapes[0];
    this.lineWidth = 2;
    this.lineType = _VectorLayerEditor.lineTypes[0];
    this.fillOpacity = 1;
    this.shouldStyleByAttribute = false;
    this.fields = [];
    this.selectedField = "";
    this.fieldValues = [];
    this.selectedFieldValue = "";
  }
  static get componentName() {
    return `vector-layer-editor`;
  }
  static {
    this.styles = [
      flexStyles,
      legacyStyles,
      i`
            .style-by-attribute-checkbox {
                vertical-align: middle;
            }
        `
    ];
  }
  static {
    this.pointShapes = [
      "circle",
      "square",
      "diamond",
      "cross",
      "plus",
      "pentagram",
      "hexagram",
      "triangle",
      "triangle_up",
      "triangle_down",
      "triangle_left",
      "triangle_right",
      "pentagon",
      "hexagon",
      "star5",
      "star6"
    ];
  }
  static {
    this.lineTypes = [
      "solid",
      "dotted",
      "dashed"
    ];
  }
  getVisualizationOptions() {
    let visParams = {
      layerName: this.newLayerName,
      color: this.colorPicker.value,
      opacity: this.opacity,
      pointSize: this.pointSize,
      pointShape: this.pointShape,
      lineWidth: this.lineWidth,
      lineType: this.lineType,
      fillColor: this.fillColorPicker.value,
      fillOpacity: this.fillOpacity,
      shouldStyleByAttribute: this.shouldStyleByAttribute
    };
    if (this.shouldStyleByAttribute) {
      visParams.palette = this.paletteEditor?.paletteTokens ?? [];
      visParams.field = this.selectedField;
    }
    if (this.legendCustomization) {
      visParams.legend = this.legendCustomization.getLegendData();
    }
    return visParams;
  }
  render() {
    return ke`
            <div class="vertical-flex">
                <div class="horizontal-flex">
                    <span class="legacy-text">New layer name:</span>
                    <input
                        type="text"
                        class="legacy-text-input"
                        id="new-layer-name"
                        name="new-layer-name"
                        .value="${this.newLayerName}"
                        @change="${this.onNewLayerNameChanged}"
                    />
                </div>
                <div class="horizontal-flex">
                    <color-picker id="color-picker"></color-picker>
                    <span class="legacy-text">Opacity:</span>
                    <input
                        type="range"
                        class="legacy-slider"
                        id="opacity"
                        name="opacity"
                        min="0"
                        max="1.0"
                        step="0.01"
                        .value=${this.opacity}
                        @input=${this.onOpacityChanged}
                    />
                    <span class="legacy-text">${this.opacity.toFixed(2)}</span>
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Point size:</span>
                    <input
                        type="number"
                        class="legacy-text-input"
                        id="point-size"
                        name="point-size"
                        min="1"
                        max="50"
                        .value="${this.pointSize}"
                        @change="${this.onPointSizeChanged}"
                    />
                    <span class="legacy-text">Point shape:</span>
                    ${renderSelect(_VectorLayerEditor.pointShapes, this.pointShape, this.onPointShapeChanged)}
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Line width:</span>
                    <input
                        type="number"
                        class="legacy-text-input"
                        id="line-width"
                        name="line-width"
                        min="1"
                        max="50"
                        .value="${this.lineWidth}"
                        @change="${this.onLineWidthChanged}"
                    />
                    <span class="legacy-text">Line type:</span>
                    ${renderSelect(_VectorLayerEditor.lineTypes, this.lineType, this.onLineTypeChanged)}
                </div>
                <div class="horizontal-flex">
                    <span class="legacy-text">Fill:</span>
                    <color-picker id="fill-color-picker"></color-picker>
                    <span class="legacy-text">Opacity:</span>
                    <input
                        type="range"
                        class="legacy-slider"
                        id="fill-opacity"
                        name="fill-opacity"
                        min="0"
                        max="1.0"
                        step="0.01"
                        .value=${this.fillOpacity}
                        @input=${this.onFillOpacityChanged}
                    />
                    <span class="legacy-text">${this.fillOpacity.toFixed(2)}</span>
                </div>
                <div class="horizontal-flex">
                    <span>
                        <input
                            class="style-by-attribute-checkbox"
                            type="checkbox"
                            .checked="${this.shouldStyleByAttribute}"
                            @change="${this.onShouldStyleByAttributeChanged}"
                        />
                        <span class="legacy-text all-layers-text"
                            >Style by attribute</span
                        >
                    </span>
                </div>
                ${this.renderStyleByAttributeSettings()}
                ${this.renderLegendCustomization()}
            </div>
        `;
  }
  renderStyleByAttributeSettings() {
    if (!this.shouldStyleByAttribute) {
      return D;
    }
    return ke`
            <div class="horizontal-flex">
                <span class="legacy-text">Field:</span>
                ${renderSelect(this.fields, this.selectedField, this.onStyleAttributeChanged)}
                <span class="legacy-text">Values:</span>
                ${renderSelect(this.fieldValues, this.selectedFieldValue, this.onValueChanged)}
            </div>
            <palette-editor .colormaps="${this.colormaps}">
                <slot></slot>
            </palette-editor>
        `;
  }
  renderLegendCustomization() {
    if (this.shouldStyleByAttribute) {
      return ke`<legend-customization></legend-customization>`;
    }
    return D;
  }
  onNewLayerNameChanged(event) {
    this.newLayerName = event.target.value;
  }
  onOpacityChanged(event) {
    this.opacity = event.target.valueAsNumber;
  }
  onPointSizeChanged(event) {
    this.pointSize = event.target.valueAsNumber;
  }
  onPointShapeChanged(event) {
    this.pointShape = event.target.value;
  }
  onLineWidthChanged(event) {
    this.lineWidth = event.target.valueAsNumber;
  }
  onLineTypeChanged(event) {
    this.lineType = event.target.value;
  }
  onFillOpacityChanged(event) {
    this.fillOpacity = event.target.valueAsNumber;
  }
  onShouldStyleByAttributeChanged(event) {
    this.shouldStyleByAttribute = event.target.checked;
    this.dispatchEvent(new CustomEvent("calculate-fields", {}));
  }
  onStyleAttributeChanged(event) {
    this.selectedField = event.target.value;
    this.dispatchEvent(new CustomEvent("calculate-field-values", {}));
  }
  onValueChanged(event) {
    this.selectedFieldValue = event.target.value;
  }
};
__decorateClass([
  n4({ type: Array })
], _VectorLayerEditor.prototype, "colormaps", 2);
__decorateClass([
  n4({ type: String })
], _VectorLayerEditor.prototype, "newLayerName", 2);
__decorateClass([
  n4({ type: Number })
], _VectorLayerEditor.prototype, "opacity", 2);
__decorateClass([
  n4({ type: Number })
], _VectorLayerEditor.prototype, "pointSize", 2);
__decorateClass([
  n4({ type: String })
], _VectorLayerEditor.prototype, "pointShape", 2);
__decorateClass([
  n4({ type: Number })
], _VectorLayerEditor.prototype, "lineWidth", 2);
__decorateClass([
  n4({ type: String })
], _VectorLayerEditor.prototype, "lineType", 2);
__decorateClass([
  n4({ type: Number })
], _VectorLayerEditor.prototype, "fillOpacity", 2);
__decorateClass([
  n4({ type: Boolean })
], _VectorLayerEditor.prototype, "shouldStyleByAttribute", 2);
__decorateClass([
  n4({ type: Array })
], _VectorLayerEditor.prototype, "fields", 2);
__decorateClass([
  n4({ type: String })
], _VectorLayerEditor.prototype, "selectedField", 2);
__decorateClass([
  n4({ type: Array })
], _VectorLayerEditor.prototype, "fieldValues", 2);
__decorateClass([
  n4({ type: String })
], _VectorLayerEditor.prototype, "selectedFieldValue", 2);
__decorateClass([
  e4("palette-editor")
], _VectorLayerEditor.prototype, "paletteEditor", 2);
__decorateClass([
  e4("legend-customization")
], _VectorLayerEditor.prototype, "legendCustomization", 2);
__decorateClass([
  e4("#color-picker")
], _VectorLayerEditor.prototype, "colorPicker", 2);
__decorateClass([
  e4("#fill-color-picker")
], _VectorLayerEditor.prototype, "fillColorPicker", 2);
var VectorLayerEditor = _VectorLayerEditor;
if (!customElements.get(VectorLayerEditor.componentName)) {
  customElements.define(VectorLayerEditor.componentName, VectorLayerEditor);
}

// js/layer_editor.ts
var LayerEditor = class extends LitWidget {
  constructor() {
    super(...arguments);
    this.layerName = "";
    this.layerType = "";
    this.bandNames = [];
    this.colormaps = [];
  }
  static get componentName() {
    return `layer-editor`;
  }
  static {
    this.styles = [
      legacyStyles,
      i`
            .confirm-button {
                padding: 0 20px;
            }

            .confirm-button-row {
                display: flex;
                gap: 4px;
                margin-top: 4px;
            }

            .editor-container {
                max-height: 250px;
                max-width: 350px;
                overflow-y: auto;
            }
        `
    ];
  }
  modelNameToViewName() {
    return /* @__PURE__ */ new Map([
      ["layer_name", "layerName"],
      ["layer_type", "layerType"],
      ["band_names", "bandNames"],
      ["colormaps", "colormaps"],
      ["children", null]
    ]);
  }
  onCustomMessage(msg) {
    const msgId = msg.id;
    const response = msg.response;
    if (msgId === "band-stats") {
      this.handleBandStatsResponse(response);
    } else if (msgId === "palette") {
      this.handlePaletteResponse(response);
    } else if (msgId === "fields") {
      this.handleFieldResponse(response);
    } else if (msgId === "field-values") {
      this.handleFieldValuesResponse(response);
    }
  }
  render() {
    return ke`
            <widget-container
                .title="${this.layerName}"
                @close-clicked="${this.onCloseButtonClicked}"
            >
                <div class="editor-container">
                    ${this.renderLayerEditorType()}
                </div>
                <div class="confirm-button-row">
                    <button
                        class="legacy-button primary confirm-button"
                        @click="${this.onImportClicked}"
                    >
                        Import
                    </button>
                    <button
                        class="legacy-button confirm-button"
                        @click="${this.onApplyClicked}"
                    >
                        Apply
                    </button>
                </div>
            </widget-container>
        `;
  }
  renderLayerEditorType() {
    if (this.layerType == "raster") {
      return ke`
                <raster-layer-editor
                    .bandNames="${this.bandNames}"
                    .colormaps="${this.colormaps}"
                    @calculate-band-stats="${this.calculateBandStats}"
                    @calculate-palette="${this.calculatePalette}"
                >
                    <slot></slot>
                </raster-layer-editor>
            `;
    } else if (this.layerType == "vector") {
      return ke`
                <vector-layer-editor
                    .newLayerName="${this.layerName + " style"}"
                    .colormaps="${this.colormaps}"
                    @calculate-fields="${this.calculateFields}"
                    @calculate-field-values="${this.calculateFieldValues}"
                    @calculate-palette="${this.calculatePalette}"
                >
                    <slot></slot>
                </vector-layer-editor>
            `;
    }
    return ke`<div><span>Vis params are uneditable</span></div>`;
  }
  onCloseButtonClicked(_2) {
    this.model?.send({ type: "click", id: "close" });
  }
  onApplyClicked(_event) {
    this.sendCompletion("apply");
  }
  onImportClicked(_event) {
    this.sendCompletion("import");
  }
  sendCompletion(completionType) {
    if (this.rasterEditor) {
      this.model?.send({
        id: completionType,
        type: "click",
        detail: this.rasterEditor.getVisualizationOptions()
      });
    } else if (this.vectorEditor) {
      this.model?.send({
        id: completionType,
        type: "click",
        detail: this.vectorEditor.getVisualizationOptions()
      });
    }
  }
  calculateBandStats(_event) {
    if (this.rasterEditor) {
      this.model?.send({
        id: "band-stats",
        type: "calculate",
        detail: {
          bands: this.rasterEditor.selectedBands,
          stretch: this.rasterEditor.stretch
        }
      });
    }
  }
  handleBandStatsResponse(response) {
    if (this.rasterEditor) {
      if (response.stretch === this.rasterEditor.stretch) {
        this.rasterEditor.minValue = response.min;
        this.rasterEditor.maxValue = response.max;
      }
    }
  }
  calculatePalette(event) {
    this.model?.send({
      id: "palette",
      type: "calculate",
      detail: {
        colormap: event.detail.colormap,
        classes: event.detail.classes,
        palette: event.detail.palette,
        bandMin: this.rasterEditor?.minValue ?? 0,
        bandMax: this.rasterEditor?.maxValue ?? 1
      }
    });
  }
  getLegendClassLabels(palette) {
    if (palette === "") {
      return [];
    }
    const length = palette.split(",").length;
    return Array.from({ length }, (_2, i4) => `Class ${i4 + 1}`);
  }
  handlePaletteResponse(response) {
    const paletteEditor = (this.rasterEditor || this.vectorEditor)?.paletteEditor;
    const legendCustomization = (this.rasterEditor || this.vectorEditor)?.legendCustomization;
    if (paletteEditor && response.palette) {
      paletteEditor.palette = response.palette;
    }
    if (legendCustomization) {
      legendCustomization.labels = this.getLegendClassLabels(response.palette);
    }
  }
  calculateFields(_event) {
    this.model?.send({ id: "fields", type: "calculate", detail: {} });
  }
  handleFieldResponse(response) {
    if (this.vectorEditor) {
      const fields = response.fields;
      const values = response["field-values"];
      this.vectorEditor.fields = fields;
      this.vectorEditor.fieldValues = values;
      this.vectorEditor.selectedField = fields.length > 0 ? fields[0] : "";
      this.vectorEditor.selectedFieldValue = values.length > 0 ? values[0] : "";
    }
  }
  calculateFieldValues(_event) {
    if (this.vectorEditor) {
      this.model?.send({
        id: "field-values",
        type: "calculate",
        detail: {
          field: this.vectorEditor?.selectedField
        }
      });
    }
  }
  handleFieldValuesResponse(response) {
    if (this.vectorEditor) {
      const values = response["field-values"];
      this.vectorEditor.fieldValues = values;
      this.vectorEditor.selectedFieldValue = values.length > 0 ? values[0] : "";
    }
  }
};
__decorateClass([
  n4({ type: String })
], LayerEditor.prototype, "layerName", 2);
__decorateClass([
  n4({ type: String })
], LayerEditor.prototype, "layerType", 2);
__decorateClass([
  n4({ type: Array })
], LayerEditor.prototype, "bandNames", 2);
__decorateClass([
  n4({ type: Array })
], LayerEditor.prototype, "colormaps", 2);
__decorateClass([
  e4("raster-layer-editor")
], LayerEditor.prototype, "rasterEditor", 2);
__decorateClass([
  e4("vector-layer-editor")
], LayerEditor.prototype, "vectorEditor", 2);
if (!customElements.get(LayerEditor.componentName)) {
  customElements.define(LayerEditor.componentName, LayerEditor);
}
async function render({ model, el }) {
  const widget = document.createElement(LayerEditor.componentName);
  widget.model = model;
  el.appendChild(widget);
  updateChildren(widget, model);
  model.on("change:children", () => {
    updateChildren(widget, model);
  });
}
var layer_editor_default = { render };
export {
  LayerEditor,
  layer_editor_default as default
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
*/
