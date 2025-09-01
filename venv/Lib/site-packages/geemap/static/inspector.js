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
  constructor(t3, e5, o4) {
    if (this._$cssResult$ = true, o4 !== s) throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");
    this.cssText = t3, this.t = e5;
  }
  get styleSheet() {
    let t3 = this.o;
    const s2 = this.t;
    if (e && void 0 === t3) {
      const e5 = void 0 !== s2 && 1 === s2.length;
      e5 && (t3 = o.get(s2)), void 0 === t3 && ((this.o = t3 = new CSSStyleSheet()).replaceSync(this.cssText), e5 && o.set(s2, t3));
    }
    return t3;
  }
  toString() {
    return this.cssText;
  }
};
var r = (t3) => new n("string" == typeof t3 ? t3 : t3 + "", void 0, s);
var i = (t3, ...e5) => {
  const o4 = 1 === t3.length ? t3[0] : e5.reduce((e6, s2, o5) => e6 + ((t4) => {
    if (true === t4._$cssResult$) return t4.cssText;
    if ("number" == typeof t4) return t4;
    throw Error("Value passed to 'css' function must be a 'css' function result: " + t4 + ". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.");
  })(s2) + t3[o5 + 1], t3[0]);
  return new n(o4, t3, s);
};
var S = (s2, o4) => {
  if (e) s2.adoptedStyleSheets = o4.map((t3) => t3 instanceof CSSStyleSheet ? t3 : t3.styleSheet);
  else for (const e5 of o4) {
    const o5 = document.createElement("style"), n5 = t.litNonce;
    void 0 !== n5 && o5.setAttribute("nonce", n5), o5.textContent = e5.cssText, s2.appendChild(o5);
  }
};
var c = e ? (t3) => t3 : (t3) => t3 instanceof CSSStyleSheet ? ((t4) => {
  let e5 = "";
  for (const s2 of t4.cssRules) e5 += s2.cssText;
  return r(e5);
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
      const i4 = Symbol(), r4 = this.getPropertyDescriptor(t3, i4, s2);
      void 0 !== r4 && e2(this.prototype, t3, r4);
    }
  }
  static getPropertyDescriptor(t3, s2, i4) {
    const { get: e5, set: h4 } = r2(this.prototype, t3) ?? { get() {
      return this[s2];
    }, set(t4) {
      this[s2] = t4;
    } };
    return { get() {
      return e5?.call(this);
    }, set(s3) {
      const r4 = e5?.call(this);
      h4.call(this, s3), this.requestUpdate(t3, r4, i4);
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
      const e5 = new Set(s2.flat(1 / 0).reverse());
      for (const s3 of e5) i4.unshift(c(s3));
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
    const i4 = this.constructor.elementProperties.get(t3), e5 = this.constructor._$Eu(t3, i4);
    if (void 0 !== e5 && true === i4.reflect) {
      const r4 = (void 0 !== i4.converter?.toAttribute ? i4.converter : u).toAttribute(s2, i4.type);
      this._$Em = t3, null == r4 ? this.removeAttribute(e5) : this.setAttribute(e5, r4), this._$Em = null;
    }
  }
  _$AK(t3, s2) {
    const i4 = this.constructor, e5 = i4._$Eh.get(t3);
    if (void 0 !== e5 && this._$Em !== e5) {
      const t4 = i4.getPropertyOptions(e5), r4 = "function" == typeof t4.converter ? { fromAttribute: t4.converter } : void 0 !== t4.converter?.fromAttribute ? t4.converter : u;
      this._$Em = e5, this[e5] = r4.fromAttribute(s2, t4.type), this._$Em = null;
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
  const s2 = t3.length - 1, e5 = [];
  let h4, o4 = 2 === i4 ? "<svg>" : 3 === i4 ? "<math>" : "", n5 = T;
  for (let i5 = 0; i5 < s2; i5++) {
    const s3 = t3[i5];
    let r4, l2, c4 = -1, a2 = 0;
    for (; a2 < s3.length && (n5.lastIndex = a2, l2 = n5.exec(s3), null !== l2); ) a2 = n5.lastIndex, n5 === T ? "!--" === l2[1] ? n5 = E : void 0 !== l2[1] ? n5 = k : void 0 !== l2[2] ? (M.test(l2[2]) && (h4 = RegExp("</" + l2[2], "g")), n5 = O) : void 0 !== l2[3] && (n5 = O) : n5 === O ? ">" === l2[0] ? (n5 = h4 ?? T, c4 = -1) : void 0 === l2[1] ? c4 = -2 : (c4 = n5.lastIndex - l2[2].length, r4 = l2[1], n5 = void 0 === l2[3] ? O : '"' === l2[3] ? j : S2) : n5 === j || n5 === S2 ? n5 = O : n5 === E || n5 === k ? n5 = T : (n5 = O, h4 = void 0);
    const u2 = n5 === O && t3[i5 + 1].startsWith("/>") ? " " : "";
    o4 += n5 === T ? s3 + _ : c4 >= 0 ? (e5.push(r4), s3.slice(0, c4) + f2 + s3.slice(c4) + v + u2) : s3 + v + (-2 === c4 ? i5 : u2);
  }
  return [N(t3, o4 + (t3[s2] || "<?>") + (2 === i4 ? "</svg>" : 3 === i4 ? "</math>" : "")), e5];
};
var B = class _B {
  constructor({ strings: t3, _$litType$: i4 }, s2) {
    let e5;
    this.parts = [];
    let h4 = 0, o4 = 0;
    const n5 = t3.length - 1, r4 = this.parts, [l2, a2] = U(t3, i4);
    if (this.el = _B.createElement(l2, s2), I.currentNode = this.el.content, 2 === i4 || 3 === i4) {
      const t4 = this.el.content.firstChild;
      t4.replaceWith(...t4.childNodes);
    }
    for (; null !== (e5 = I.nextNode()) && r4.length < n5; ) {
      if (1 === e5.nodeType) {
        if (e5.hasAttributes()) for (const t4 of e5.getAttributeNames()) if (t4.endsWith(f2)) {
          const i5 = a2[o4++], s3 = e5.getAttribute(t4).split(v), n6 = /([.?@])?(.*)/.exec(i5);
          r4.push({ type: 1, index: h4, name: n6[2], strings: s3, ctor: "." === n6[1] ? Y : "?" === n6[1] ? Z : "@" === n6[1] ? q : G }), e5.removeAttribute(t4);
        } else t4.startsWith(v) && (r4.push({ type: 6, index: h4 }), e5.removeAttribute(t4));
        if (M.test(e5.tagName)) {
          const t4 = e5.textContent.split(v), i5 = t4.length - 1;
          if (i5 > 0) {
            e5.textContent = c3 ? c3.emptyScript : "";
            for (let s3 = 0; s3 < i5; s3++) e5.append(t4[s3], lt()), I.nextNode(), r4.push({ type: 2, index: ++h4 });
            e5.append(t4[i5], lt());
          }
        }
      } else if (8 === e5.nodeType) if (e5.data === m) r4.push({ type: 2, index: h4 });
      else {
        let t4 = -1;
        for (; -1 !== (t4 = e5.data.indexOf(v, t4 + 1)); ) r4.push({ type: 7, index: h4 }), t4 += v.length - 1;
      }
      h4++;
    }
  }
  static createElement(t3, i4) {
    const s2 = w.createElement("template");
    return s2.innerHTML = t3, s2;
  }
};
function z(t3, i4, s2 = t3, e5) {
  if (i4 === R) return i4;
  let h4 = void 0 !== e5 ? s2.o?.[e5] : s2.l;
  const o4 = st(i4) ? void 0 : i4._$litDirective$;
  return h4?.constructor !== o4 && (h4?._$AO?.(false), void 0 === o4 ? h4 = void 0 : (h4 = new o4(t3), h4._$AT(t3, s2, e5)), void 0 !== e5 ? (s2.o ??= [])[e5] = h4 : s2.l = h4), void 0 !== h4 && (i4 = z(t3, h4._$AS(t3, i4.values), h4, e5)), i4;
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
    const { el: { content: i4 }, parts: s2 } = this._$AD, e5 = (t3?.creationScope ?? w).importNode(i4, true);
    I.currentNode = e5;
    let h4 = I.nextNode(), o4 = 0, n5 = 0, r4 = s2[0];
    for (; void 0 !== r4; ) {
      if (o4 === r4.index) {
        let i5;
        2 === r4.type ? i5 = new et(h4, h4.nextSibling, this, t3) : 1 === r4.type ? i5 = new r4.ctor(h4, r4.name, r4.strings, this, t3) : 6 === r4.type && (i5 = new K(h4, this, t3)), this._$AV.push(i5), r4 = s2[++n5];
      }
      o4 !== r4?.index && (h4 = I.nextNode(), o4++);
    }
    return I.currentNode = w, e5;
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
  constructor(t3, i4, s2, e5) {
    this.type = 2, this._$AH = D, this._$AN = void 0, this._$AA = t3, this._$AB = i4, this._$AM = s2, this.options = e5, this.v = e5?.isConnected ?? true;
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
    const { values: i4, _$litType$: s2 } = t3, e5 = "number" == typeof s2 ? this._$AC(t3) : (void 0 === s2.el && (s2.el = B.createElement(N(s2.h, s2.h[0]), this.options)), s2);
    if (this._$AH?._$AD === e5) this._$AH.p(i4);
    else {
      const t4 = new F(e5, this), s3 = t4.u(this.options);
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
    let s2, e5 = 0;
    for (const h4 of t3) e5 === i4.length ? i4.push(s2 = new _et(this.O(lt()), this.O(lt()), this, this.options)) : s2 = i4[e5], s2._$AI(h4), e5++;
    e5 < i4.length && (this._$AR(s2 && s2._$AB.nextSibling, e5), i4.length = e5);
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
  constructor(t3, i4, s2, e5, h4) {
    this.type = 1, this._$AH = D, this._$AN = void 0, this.element = t3, this.name = i4, this._$AM = e5, this.options = h4, s2.length > 2 || "" !== s2[0] || "" !== s2[1] ? (this._$AH = Array(s2.length - 1).fill(new String()), this.strings = s2) : this._$AH = D;
  }
  _$AI(t3, i4 = this, s2, e5) {
    const h4 = this.strings;
    let o4 = false;
    if (void 0 === h4) t3 = z(this, t3, i4, 0), o4 = !st(t3) || t3 !== this._$AH && t3 !== R, o4 && (this._$AH = t3);
    else {
      const e6 = t3;
      let n5, r4;
      for (t3 = h4[0], n5 = 0; n5 < h4.length - 1; n5++) r4 = z(this, e6[s2 + n5], i4, n5), r4 === R && (r4 = this._$AH[n5]), o4 ||= !st(r4) || r4 !== this._$AH[n5], r4 === D ? t3 = D : t3 !== D && (t3 += (r4 ?? "") + h4[n5 + 1]), this._$AH[n5] = r4;
    }
    o4 && !e5 && this.j(t3);
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
  constructor(t3, i4, s2, e5, h4) {
    super(t3, i4, s2, e5, h4), this.type = 5;
  }
  _$AI(t3, i4 = this) {
    if ((t3 = z(this, t3, i4, 0) ?? D) === R) return;
    const s2 = this._$AH, e5 = t3 === D && s2 !== D || t3.capture !== s2.capture || t3.once !== s2.once || t3.passive !== s2.passive, h4 = t3 !== D && (s2 === D || e5);
    e5 && this.element.removeEventListener(this.name, this, s2), h4 && this.element.addEventListener(this.name, this, t3), this._$AH = t3;
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
  const e5 = s2?.renderBefore ?? i4;
  let h4 = e5._$litPart$;
  if (void 0 === h4) {
    const t4 = s2?.renderBefore ?? null;
    e5._$litPart$ = h4 = new et(i4.insertBefore(lt(), t4), t4, void 0, s2 ?? {});
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
    const e5 = this.render();
    this.hasUpdated || (this.renderOptions.isConnected = this.isConnected), super.update(t3), this.o = Q(e5, this.renderRoot, this.renderOptions);
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
var r3 = (t3 = o3, e5, r4) => {
  const { kind: n5, metadata: i4 } = r4;
  let s2 = globalThis.litPropertyMetadata.get(i4);
  if (void 0 === s2 && globalThis.litPropertyMetadata.set(i4, s2 = /* @__PURE__ */ new Map()), s2.set(r4.name, t3), "accessor" === n5) {
    const { name: o4 } = r4;
    return { set(r5) {
      const n6 = e5.get.call(this);
      e5.set.call(this, r5), this.requestUpdate(o4, n6, t3);
    }, init(e6) {
      return void 0 !== e6 && this.P(o4, void 0, t3), e6;
    } };
  }
  if ("setter" === n5) {
    const { name: o4 } = r4;
    return function(r5) {
      const n6 = this[o4];
      e5.call(this, r5), this.requestUpdate(o4, n6, t3);
    };
  }
  throw Error("Unsupported decorator location: " + n5);
};
function n4(t3) {
  return (e5, o4) => "object" == typeof o4 ? r3(t3, e5, o4) : ((t4, e6, o5) => {
    const r4 = e6.hasOwnProperty(o5);
    return e6.constructor.createProperty(o5, r4 ? { ...t4, wrapped: true } : t4), r4 ? Object.getOwnPropertyDescriptor(e6, o5) : void 0;
  })(t3, e5, o4);
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

// node_modules/lit-html/directive.js
var t2 = { ATTRIBUTE: 1, CHILD: 2, PROPERTY: 3, BOOLEAN_ATTRIBUTE: 4, EVENT: 5, ELEMENT: 6 };
var e4 = (t3) => (...e5) => ({ _$litDirective$: t3, values: e5 });
var i3 = class {
  constructor(t3) {
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  _$AT(t3, e5, i4) {
    this.t = t3, this._$AM = e5, this.i = i4;
  }
  _$AS(t3, e5) {
    return this.update(t3, e5);
  }
  update(t3, e5) {
    return this.render(...e5);
  }
};

// node_modules/lit-html/directives/class-map.js
var Rt = e4(class extends i3 {
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
      const r4 = !!s2[t4];
      r4 === this.st.has(t4) || this.nt?.has(t4) || (r4 ? (i4.add(t4), this.st.add(t4)) : (i4.remove(t4), this.st.delete(t4)));
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

// js/tree_node.ts
var TreeNode = class extends h3 {
  constructor() {
    super(...arguments);
    this.node = {};
    this.expanded = false;
  }
  static get componentName() {
    return `tree-node`;
  }
  static {
    this.styles = [
      legacyStyles,
      materialStyles,
      i`
            .node {
                align-items: center;
                cursor: pointer;
                display: flex;
            }

            .node-text {
                height: auto;
                line-height: 24px;
            }

            .node:hover {
                background-color: var(--jp-layout-color2);
                margin-left: -100%;
                padding-left: 100%;
            }

            .icon {
                font-size: 13px;
                width: 20px;
            }

            ul {
                list-style: none;
                padding-left: 20px;
                margin: 0;
            }
        `
    ];
  }
  updated(changedProperties) {
    super.updated(changedProperties);
    if (changedProperties.has("node") && this.node) {
      if ("expanded" in this.node) {
        this.expanded = this.node.expanded ?? false;
      }
    }
  }
  render() {
    return ke`
            <div
                class="node ${this.expanded ? "expanded" : ""}"
                @click="${this.toggleExpand}"
            >
                ${this.renderBullet()} ${this.renderIcon()}
                <span class="legacy-text node-text">${this.node.label}</span>
            </div>
            ${this.renderChildren()}
        `;
  }
  toggleExpand() {
    this.expanded = !this.expanded;
  }
  hasChildren() {
    return !!this.node.children?.length;
  }
  renderChildren() {
    if (this.expanded && this.hasChildren()) {
      return ke`<ul>${this.node.children?.map(this.renderChild)}</ul>`;
    }
    return D;
  }
  renderChild(child) {
    return ke`<li><tree-node .node="${child}"></tree-node></li>`;
  }
  renderBullet() {
    if (this.node.topLevel) {
      if (this.expanded) {
        return ke`
                    <span class="icon material-symbols-outlined"
                        >indeterminate_check_box</span
                    >
                `;
      }
      return ke`<span class="icon material-symbols-outlined">add_box</span>`;
    } else if (this.hasChildren()) {
      if (this.expanded) {
        return ke`<span class="icon material-symbols-outlined">remove</span>`;
      }
      return ke`<span class="icon material-symbols-outlined">add</span>`;
    }
    return ke`<span class="icon"></span>`;
  }
  renderIcon() {
    if (this.node.topLevel) {
      return ke`<span class="icon material-symbols-outlined"
                >inventory_2</span
            >`;
    } else if (this.hasChildren()) {
      if (this.expanded) {
        return ke`
                    <span class="icon material-symbols-outlined">folder_open</span>
                `;
      }
      return ke`<span class="icon material-symbols-outlined">folder</span>`;
    }
    return ke`<span class="icon material-symbols-outlined">draft</span>`;
  }
};
__decorateClass([
  n4()
], TreeNode.prototype, "node", 2);
__decorateClass([
  n4({ type: Boolean, reflect: true })
], TreeNode.prototype, "expanded", 2);
if (!customElements.get(TreeNode.componentName)) {
  customElements.define(TreeNode.componentName, TreeNode);
}

// js/inspector.ts
var Inspector = class extends LitWidget {
  constructor() {
    super(...arguments);
    this.hideCloseButton = false;
    this.expandPoints = false;
    this.expandPixels = true;
    this.expandObjects = false;
    this.pointInfo = {};
    this.pixelInfo = {};
    this.objectInfo = {};
  }
  static get componentName() {
    return `inspector-widget`;
  }
  static {
    this.styles = [
      legacyStyles,
      i`
            .checkbox-container {
                align-items: center;
                display: flex;
                gap: 8px;
                height: 32px;
            }

            .spacer {
                width: 8px;
            }

            .object-browser {
                max-height: 300px;
                overflow: auto;
                width: 290px;
            }

            input[type='checkbox'] {
                vertical-align: middle;
            }
        `
    ];
  }
  modelNameToViewName() {
    return /* @__PURE__ */ new Map([
      ["hide_close_button", "hideCloseButton"],
      ["expand_points", "expandPoints"],
      ["expand_pixels", "expandPixels"],
      ["expand_objects", "expandObjects"],
      ["point_info", "pointInfo"],
      ["pixel_info", "pixelInfo"],
      ["object_info", "objectInfo"]
    ]);
  }
  render() {
    return ke`
            <widget-container
                icon="point_scan"
                title="Inspector"
                .hideCloseButton="${this.hideCloseButton}"
                @close-clicked="${this.onCloseButtonClicked}"
            >
                <div class="checkbox-container">
                    <span class="legacy-text">Expand</span>
                    <div>
                        <input
                            type="checkbox"
                            .checked="${this.expandPoints}"
                            @change="${this.onPointCheckboxEvent}"
                        />
                        <span class="legacy-text">Point</span>
                    </div>
                    <div>
                        <input
                            type="checkbox"
                            .checked="${this.expandPixels}"
                            @change="${this.onPixelCheckboxEvent}"
                        />
                        <span class="legacy-text">Pixels</span>
                    </div>
                    <div>
                        <input
                            type="checkbox"
                            .checked="${this.expandObjects}"
                            @change="${this.onFeatureCheckboxEvent}"
                        />
                        <span class="legacy-text">Objects</span>
                    </div>
                </div>
                <div class="object-browser">
                    ${this.renderNode(this.pointInfo)}
                    ${this.renderNode(this.pixelInfo)}
                    ${this.renderNode(this.objectInfo)}
                </div>
            </widget-container>
        `;
  }
  renderNode(node) {
    if (node.children?.length) {
      return ke`<tree-node .node="${node}"></tree-node> `;
    }
    return D;
  }
  onPointCheckboxEvent(event) {
    const target = event.target;
    this.expandPoints = target.checked;
  }
  onPixelCheckboxEvent(event) {
    const target = event.target;
    this.expandPixels = target.checked;
  }
  onFeatureCheckboxEvent(event) {
    const target = event.target;
    this.expandObjects = target.checked;
  }
  onCloseButtonClicked(_2) {
    this.model?.send({ "type": "click", "id": "close" });
  }
};
__decorateClass([
  n4({ type: Boolean })
], Inspector.prototype, "hideCloseButton", 2);
__decorateClass([
  n4({ type: Boolean })
], Inspector.prototype, "expandPoints", 2);
__decorateClass([
  n4({ type: Boolean })
], Inspector.prototype, "expandPixels", 2);
__decorateClass([
  n4({ type: Boolean })
], Inspector.prototype, "expandObjects", 2);
__decorateClass([
  n4()
], Inspector.prototype, "pointInfo", 2);
__decorateClass([
  n4()
], Inspector.prototype, "pixelInfo", 2);
__decorateClass([
  n4()
], Inspector.prototype, "objectInfo", 2);
if (!customElements.get(Inspector.componentName)) {
  customElements.define(Inspector.componentName, Inspector);
}
async function render({ model, el }) {
  const widget = document.createElement(Inspector.componentName);
  widget.model = model;
  el.appendChild(widget);
}
var inspector_default = { render };
export {
  Inspector,
  inspector_default as default
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
