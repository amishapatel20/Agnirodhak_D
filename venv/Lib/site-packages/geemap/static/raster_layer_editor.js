var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __decorateClass = (decorators, target, key, kind) => {
  var result = kind > 1 ? void 0 : kind ? __getOwnPropDesc(target, key) : target;
  for (var i3 = decorators.length - 1, decorator; i3 >= 0; i3--)
    if (decorator = decorators[i3])
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
  constructor(t2, e6, o4) {
    if (this._$cssResult$ = true, o4 !== s) throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");
    this.cssText = t2, this.t = e6;
  }
  get styleSheet() {
    let t2 = this.o;
    const s2 = this.t;
    if (e && void 0 === t2) {
      const e6 = void 0 !== s2 && 1 === s2.length;
      e6 && (t2 = o.get(s2)), void 0 === t2 && ((this.o = t2 = new CSSStyleSheet()).replaceSync(this.cssText), e6 && o.set(s2, t2));
    }
    return t2;
  }
  toString() {
    return this.cssText;
  }
};
var r = (t2) => new n("string" == typeof t2 ? t2 : t2 + "", void 0, s);
var i = (t2, ...e6) => {
  const o4 = 1 === t2.length ? t2[0] : e6.reduce((e7, s2, o5) => e7 + ((t3) => {
    if (true === t3._$cssResult$) return t3.cssText;
    if ("number" == typeof t3) return t3;
    throw Error("Value passed to 'css' function must be a 'css' function result: " + t3 + ". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.");
  })(s2) + t2[o5 + 1], t2[0]);
  return new n(o4, t2, s);
};
var S = (s2, o4) => {
  if (e) s2.adoptedStyleSheets = o4.map((t2) => t2 instanceof CSSStyleSheet ? t2 : t2.styleSheet);
  else for (const e6 of o4) {
    const o5 = document.createElement("style"), n5 = t.litNonce;
    void 0 !== n5 && o5.setAttribute("nonce", n5), o5.textContent = e6.cssText, s2.appendChild(o5);
  }
};
var c = e ? (t2) => t2 : (t2) => t2 instanceof CSSStyleSheet ? ((t3) => {
  let e6 = "";
  for (const s2 of t3.cssRules) e6 += s2.cssText;
  return r(e6);
})(t2) : t2;

// node_modules/@lit/reactive-element/reactive-element.js
var { is: i2, defineProperty: e2, getOwnPropertyDescriptor: r2, getOwnPropertyNames: h, getOwnPropertySymbols: o2, getPrototypeOf: n2 } = Object;
var a = globalThis;
var c2 = a.trustedTypes;
var l = c2 ? c2.emptyScript : "";
var p = a.reactiveElementPolyfillSupport;
var d = (t2, s2) => t2;
var u = { toAttribute(t2, s2) {
  switch (s2) {
    case Boolean:
      t2 = t2 ? l : null;
      break;
    case Object:
    case Array:
      t2 = null == t2 ? t2 : JSON.stringify(t2);
  }
  return t2;
}, fromAttribute(t2, s2) {
  let i3 = t2;
  switch (s2) {
    case Boolean:
      i3 = null !== t2;
      break;
    case Number:
      i3 = null === t2 ? null : Number(t2);
      break;
    case Object:
    case Array:
      try {
        i3 = JSON.parse(t2);
      } catch (t3) {
        i3 = null;
      }
  }
  return i3;
} };
var f = (t2, s2) => !i2(t2, s2);
var y = { attribute: true, type: String, converter: u, reflect: false, hasChanged: f };
Symbol.metadata ??= Symbol("metadata"), a.litPropertyMetadata ??= /* @__PURE__ */ new WeakMap();
var b = class extends HTMLElement {
  static addInitializer(t2) {
    this._$Ei(), (this.l ??= []).push(t2);
  }
  static get observedAttributes() {
    return this.finalize(), this._$Eh && [...this._$Eh.keys()];
  }
  static createProperty(t2, s2 = y) {
    if (s2.state && (s2.attribute = false), this._$Ei(), this.elementProperties.set(t2, s2), !s2.noAccessor) {
      const i3 = Symbol(), r5 = this.getPropertyDescriptor(t2, i3, s2);
      void 0 !== r5 && e2(this.prototype, t2, r5);
    }
  }
  static getPropertyDescriptor(t2, s2, i3) {
    const { get: e6, set: h4 } = r2(this.prototype, t2) ?? { get() {
      return this[s2];
    }, set(t3) {
      this[s2] = t3;
    } };
    return { get() {
      return e6?.call(this);
    }, set(s3) {
      const r5 = e6?.call(this);
      h4.call(this, s3), this.requestUpdate(t2, r5, i3);
    }, configurable: true, enumerable: true };
  }
  static getPropertyOptions(t2) {
    return this.elementProperties.get(t2) ?? y;
  }
  static _$Ei() {
    if (this.hasOwnProperty(d("elementProperties"))) return;
    const t2 = n2(this);
    t2.finalize(), void 0 !== t2.l && (this.l = [...t2.l]), this.elementProperties = new Map(t2.elementProperties);
  }
  static finalize() {
    if (this.hasOwnProperty(d("finalized"))) return;
    if (this.finalized = true, this._$Ei(), this.hasOwnProperty(d("properties"))) {
      const t3 = this.properties, s2 = [...h(t3), ...o2(t3)];
      for (const i3 of s2) this.createProperty(i3, t3[i3]);
    }
    const t2 = this[Symbol.metadata];
    if (null !== t2) {
      const s2 = litPropertyMetadata.get(t2);
      if (void 0 !== s2) for (const [t3, i3] of s2) this.elementProperties.set(t3, i3);
    }
    this._$Eh = /* @__PURE__ */ new Map();
    for (const [t3, s2] of this.elementProperties) {
      const i3 = this._$Eu(t3, s2);
      void 0 !== i3 && this._$Eh.set(i3, t3);
    }
    this.elementStyles = this.finalizeStyles(this.styles);
  }
  static finalizeStyles(s2) {
    const i3 = [];
    if (Array.isArray(s2)) {
      const e6 = new Set(s2.flat(1 / 0).reverse());
      for (const s3 of e6) i3.unshift(c(s3));
    } else void 0 !== s2 && i3.push(c(s2));
    return i3;
  }
  static _$Eu(t2, s2) {
    const i3 = s2.attribute;
    return false === i3 ? void 0 : "string" == typeof i3 ? i3 : "string" == typeof t2 ? t2.toLowerCase() : void 0;
  }
  constructor() {
    super(), this._$Ep = void 0, this.isUpdatePending = false, this.hasUpdated = false, this._$Em = null, this._$Ev();
  }
  _$Ev() {
    this._$ES = new Promise((t2) => this.enableUpdating = t2), this._$AL = /* @__PURE__ */ new Map(), this._$E_(), this.requestUpdate(), this.constructor.l?.forEach((t2) => t2(this));
  }
  addController(t2) {
    (this._$EO ??= /* @__PURE__ */ new Set()).add(t2), void 0 !== this.renderRoot && this.isConnected && t2.hostConnected?.();
  }
  removeController(t2) {
    this._$EO?.delete(t2);
  }
  _$E_() {
    const t2 = /* @__PURE__ */ new Map(), s2 = this.constructor.elementProperties;
    for (const i3 of s2.keys()) this.hasOwnProperty(i3) && (t2.set(i3, this[i3]), delete this[i3]);
    t2.size > 0 && (this._$Ep = t2);
  }
  createRenderRoot() {
    const t2 = this.shadowRoot ?? this.attachShadow(this.constructor.shadowRootOptions);
    return S(t2, this.constructor.elementStyles), t2;
  }
  connectedCallback() {
    this.renderRoot ??= this.createRenderRoot(), this.enableUpdating(true), this._$EO?.forEach((t2) => t2.hostConnected?.());
  }
  enableUpdating(t2) {
  }
  disconnectedCallback() {
    this._$EO?.forEach((t2) => t2.hostDisconnected?.());
  }
  attributeChangedCallback(t2, s2, i3) {
    this._$AK(t2, i3);
  }
  _$EC(t2, s2) {
    const i3 = this.constructor.elementProperties.get(t2), e6 = this.constructor._$Eu(t2, i3);
    if (void 0 !== e6 && true === i3.reflect) {
      const r5 = (void 0 !== i3.converter?.toAttribute ? i3.converter : u).toAttribute(s2, i3.type);
      this._$Em = t2, null == r5 ? this.removeAttribute(e6) : this.setAttribute(e6, r5), this._$Em = null;
    }
  }
  _$AK(t2, s2) {
    const i3 = this.constructor, e6 = i3._$Eh.get(t2);
    if (void 0 !== e6 && this._$Em !== e6) {
      const t3 = i3.getPropertyOptions(e6), r5 = "function" == typeof t3.converter ? { fromAttribute: t3.converter } : void 0 !== t3.converter?.fromAttribute ? t3.converter : u;
      this._$Em = e6, this[e6] = r5.fromAttribute(s2, t3.type), this._$Em = null;
    }
  }
  requestUpdate(t2, s2, i3) {
    if (void 0 !== t2) {
      if (i3 ??= this.constructor.getPropertyOptions(t2), !(i3.hasChanged ?? f)(this[t2], s2)) return;
      this.P(t2, s2, i3);
    }
    false === this.isUpdatePending && (this._$ES = this._$ET());
  }
  P(t2, s2, i3) {
    this._$AL.has(t2) || this._$AL.set(t2, s2), true === i3.reflect && this._$Em !== t2 && (this._$Ej ??= /* @__PURE__ */ new Set()).add(t2);
  }
  async _$ET() {
    this.isUpdatePending = true;
    try {
      await this._$ES;
    } catch (t3) {
      Promise.reject(t3);
    }
    const t2 = this.scheduleUpdate();
    return null != t2 && await t2, !this.isUpdatePending;
  }
  scheduleUpdate() {
    return this.performUpdate();
  }
  performUpdate() {
    if (!this.isUpdatePending) return;
    if (!this.hasUpdated) {
      if (this.renderRoot ??= this.createRenderRoot(), this._$Ep) {
        for (const [t4, s3] of this._$Ep) this[t4] = s3;
        this._$Ep = void 0;
      }
      const t3 = this.constructor.elementProperties;
      if (t3.size > 0) for (const [s3, i3] of t3) true !== i3.wrapped || this._$AL.has(s3) || void 0 === this[s3] || this.P(s3, this[s3], i3);
    }
    let t2 = false;
    const s2 = this._$AL;
    try {
      t2 = this.shouldUpdate(s2), t2 ? (this.willUpdate(s2), this._$EO?.forEach((t3) => t3.hostUpdate?.()), this.update(s2)) : this._$EU();
    } catch (s3) {
      throw t2 = false, this._$EU(), s3;
    }
    t2 && this._$AE(s2);
  }
  willUpdate(t2) {
  }
  _$AE(t2) {
    this._$EO?.forEach((t3) => t3.hostUpdated?.()), this.hasUpdated || (this.hasUpdated = true, this.firstUpdated(t2)), this.updated(t2);
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
  shouldUpdate(t2) {
    return true;
  }
  update(t2) {
    this._$Ej &&= this._$Ej.forEach((t3) => this._$EC(t3, this[t3])), this._$EU();
  }
  updated(t2) {
  }
  firstUpdated(t2) {
  }
};
b.elementStyles = [], b.shadowRootOptions = { mode: "open" }, b[d("elementProperties")] = /* @__PURE__ */ new Map(), b[d("finalized")] = /* @__PURE__ */ new Map(), p?.({ ReactiveElement: b }), (a.reactiveElementVersions ??= []).push("2.0.4");

// node_modules/lit-html/lit-html.js
var n3 = globalThis;
var c3 = n3.trustedTypes;
var h2 = c3 ? c3.createPolicy("lit-html", { createHTML: (t2) => t2 }) : void 0;
var f2 = "$lit$";
var v = `lit$${Math.random().toFixed(9).slice(2)}$`;
var m = "?" + v;
var _ = `<${m}>`;
var w = document;
var lt = () => w.createComment("");
var st = (t2) => null === t2 || "object" != typeof t2 && "function" != typeof t2;
var g = Array.isArray;
var $ = (t2) => g(t2) || "function" == typeof t2?.[Symbol.iterator];
var x = "[ 	\n\f\r]";
var T = /<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g;
var E = /-->/g;
var k = />/g;
var O = RegExp(`>|${x}(?:([^\\s"'>=/]+)(${x}*=${x}*(?:[^ 	
\f\r"'\`<>=]|("|')|))|$)`, "g");
var S2 = /'/g;
var j = /"/g;
var M = /^(?:script|style|textarea|title)$/i;
var P = (t2) => (i3, ...s2) => ({ _$litType$: t2, strings: i3, values: s2 });
var ke = P(1);
var Oe = P(2);
var Se = P(3);
var R = Symbol.for("lit-noChange");
var D = Symbol.for("lit-nothing");
var V = /* @__PURE__ */ new WeakMap();
var I = w.createTreeWalker(w, 129);
function N(t2, i3) {
  if (!g(t2) || !t2.hasOwnProperty("raw")) throw Error("invalid template strings array");
  return void 0 !== h2 ? h2.createHTML(i3) : i3;
}
var U = (t2, i3) => {
  const s2 = t2.length - 1, e6 = [];
  let h4, o4 = 2 === i3 ? "<svg>" : 3 === i3 ? "<math>" : "", n5 = T;
  for (let i4 = 0; i4 < s2; i4++) {
    const s3 = t2[i4];
    let r5, l2, c4 = -1, a2 = 0;
    for (; a2 < s3.length && (n5.lastIndex = a2, l2 = n5.exec(s3), null !== l2); ) a2 = n5.lastIndex, n5 === T ? "!--" === l2[1] ? n5 = E : void 0 !== l2[1] ? n5 = k : void 0 !== l2[2] ? (M.test(l2[2]) && (h4 = RegExp("</" + l2[2], "g")), n5 = O) : void 0 !== l2[3] && (n5 = O) : n5 === O ? ">" === l2[0] ? (n5 = h4 ?? T, c4 = -1) : void 0 === l2[1] ? c4 = -2 : (c4 = n5.lastIndex - l2[2].length, r5 = l2[1], n5 = void 0 === l2[3] ? O : '"' === l2[3] ? j : S2) : n5 === j || n5 === S2 ? n5 = O : n5 === E || n5 === k ? n5 = T : (n5 = O, h4 = void 0);
    const u2 = n5 === O && t2[i4 + 1].startsWith("/>") ? " " : "";
    o4 += n5 === T ? s3 + _ : c4 >= 0 ? (e6.push(r5), s3.slice(0, c4) + f2 + s3.slice(c4) + v + u2) : s3 + v + (-2 === c4 ? i4 : u2);
  }
  return [N(t2, o4 + (t2[s2] || "<?>") + (2 === i3 ? "</svg>" : 3 === i3 ? "</math>" : "")), e6];
};
var B = class _B {
  constructor({ strings: t2, _$litType$: i3 }, s2) {
    let e6;
    this.parts = [];
    let h4 = 0, o4 = 0;
    const n5 = t2.length - 1, r5 = this.parts, [l2, a2] = U(t2, i3);
    if (this.el = _B.createElement(l2, s2), I.currentNode = this.el.content, 2 === i3 || 3 === i3) {
      const t3 = this.el.content.firstChild;
      t3.replaceWith(...t3.childNodes);
    }
    for (; null !== (e6 = I.nextNode()) && r5.length < n5; ) {
      if (1 === e6.nodeType) {
        if (e6.hasAttributes()) for (const t3 of e6.getAttributeNames()) if (t3.endsWith(f2)) {
          const i4 = a2[o4++], s3 = e6.getAttribute(t3).split(v), n6 = /([.?@])?(.*)/.exec(i4);
          r5.push({ type: 1, index: h4, name: n6[2], strings: s3, ctor: "." === n6[1] ? Y : "?" === n6[1] ? Z : "@" === n6[1] ? q : G }), e6.removeAttribute(t3);
        } else t3.startsWith(v) && (r5.push({ type: 6, index: h4 }), e6.removeAttribute(t3));
        if (M.test(e6.tagName)) {
          const t3 = e6.textContent.split(v), i4 = t3.length - 1;
          if (i4 > 0) {
            e6.textContent = c3 ? c3.emptyScript : "";
            for (let s3 = 0; s3 < i4; s3++) e6.append(t3[s3], lt()), I.nextNode(), r5.push({ type: 2, index: ++h4 });
            e6.append(t3[i4], lt());
          }
        }
      } else if (8 === e6.nodeType) if (e6.data === m) r5.push({ type: 2, index: h4 });
      else {
        let t3 = -1;
        for (; -1 !== (t3 = e6.data.indexOf(v, t3 + 1)); ) r5.push({ type: 7, index: h4 }), t3 += v.length - 1;
      }
      h4++;
    }
  }
  static createElement(t2, i3) {
    const s2 = w.createElement("template");
    return s2.innerHTML = t2, s2;
  }
};
function z(t2, i3, s2 = t2, e6) {
  if (i3 === R) return i3;
  let h4 = void 0 !== e6 ? s2.o?.[e6] : s2.l;
  const o4 = st(i3) ? void 0 : i3._$litDirective$;
  return h4?.constructor !== o4 && (h4?._$AO?.(false), void 0 === o4 ? h4 = void 0 : (h4 = new o4(t2), h4._$AT(t2, s2, e6)), void 0 !== e6 ? (s2.o ??= [])[e6] = h4 : s2.l = h4), void 0 !== h4 && (i3 = z(t2, h4._$AS(t2, i3.values), h4, e6)), i3;
}
var F = class {
  constructor(t2, i3) {
    this._$AV = [], this._$AN = void 0, this._$AD = t2, this._$AM = i3;
  }
  get parentNode() {
    return this._$AM.parentNode;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  u(t2) {
    const { el: { content: i3 }, parts: s2 } = this._$AD, e6 = (t2?.creationScope ?? w).importNode(i3, true);
    I.currentNode = e6;
    let h4 = I.nextNode(), o4 = 0, n5 = 0, r5 = s2[0];
    for (; void 0 !== r5; ) {
      if (o4 === r5.index) {
        let i4;
        2 === r5.type ? i4 = new et(h4, h4.nextSibling, this, t2) : 1 === r5.type ? i4 = new r5.ctor(h4, r5.name, r5.strings, this, t2) : 6 === r5.type && (i4 = new K(h4, this, t2)), this._$AV.push(i4), r5 = s2[++n5];
      }
      o4 !== r5?.index && (h4 = I.nextNode(), o4++);
    }
    return I.currentNode = w, e6;
  }
  p(t2) {
    let i3 = 0;
    for (const s2 of this._$AV) void 0 !== s2 && (void 0 !== s2.strings ? (s2._$AI(t2, s2, i3), i3 += s2.strings.length - 2) : s2._$AI(t2[i3])), i3++;
  }
};
var et = class _et {
  get _$AU() {
    return this._$AM?._$AU ?? this.v;
  }
  constructor(t2, i3, s2, e6) {
    this.type = 2, this._$AH = D, this._$AN = void 0, this._$AA = t2, this._$AB = i3, this._$AM = s2, this.options = e6, this.v = e6?.isConnected ?? true;
  }
  get parentNode() {
    let t2 = this._$AA.parentNode;
    const i3 = this._$AM;
    return void 0 !== i3 && 11 === t2?.nodeType && (t2 = i3.parentNode), t2;
  }
  get startNode() {
    return this._$AA;
  }
  get endNode() {
    return this._$AB;
  }
  _$AI(t2, i3 = this) {
    t2 = z(this, t2, i3), st(t2) ? t2 === D || null == t2 || "" === t2 ? (this._$AH !== D && this._$AR(), this._$AH = D) : t2 !== this._$AH && t2 !== R && this._(t2) : void 0 !== t2._$litType$ ? this.$(t2) : void 0 !== t2.nodeType ? this.T(t2) : $(t2) ? this.k(t2) : this._(t2);
  }
  O(t2) {
    return this._$AA.parentNode.insertBefore(t2, this._$AB);
  }
  T(t2) {
    this._$AH !== t2 && (this._$AR(), this._$AH = this.O(t2));
  }
  _(t2) {
    this._$AH !== D && st(this._$AH) ? this._$AA.nextSibling.data = t2 : this.T(w.createTextNode(t2)), this._$AH = t2;
  }
  $(t2) {
    const { values: i3, _$litType$: s2 } = t2, e6 = "number" == typeof s2 ? this._$AC(t2) : (void 0 === s2.el && (s2.el = B.createElement(N(s2.h, s2.h[0]), this.options)), s2);
    if (this._$AH?._$AD === e6) this._$AH.p(i3);
    else {
      const t3 = new F(e6, this), s3 = t3.u(this.options);
      t3.p(i3), this.T(s3), this._$AH = t3;
    }
  }
  _$AC(t2) {
    let i3 = V.get(t2.strings);
    return void 0 === i3 && V.set(t2.strings, i3 = new B(t2)), i3;
  }
  k(t2) {
    g(this._$AH) || (this._$AH = [], this._$AR());
    const i3 = this._$AH;
    let s2, e6 = 0;
    for (const h4 of t2) e6 === i3.length ? i3.push(s2 = new _et(this.O(lt()), this.O(lt()), this, this.options)) : s2 = i3[e6], s2._$AI(h4), e6++;
    e6 < i3.length && (this._$AR(s2 && s2._$AB.nextSibling, e6), i3.length = e6);
  }
  _$AR(t2 = this._$AA.nextSibling, i3) {
    for (this._$AP?.(false, true, i3); t2 && t2 !== this._$AB; ) {
      const i4 = t2.nextSibling;
      t2.remove(), t2 = i4;
    }
  }
  setConnected(t2) {
    void 0 === this._$AM && (this.v = t2, this._$AP?.(t2));
  }
};
var G = class {
  get tagName() {
    return this.element.tagName;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  constructor(t2, i3, s2, e6, h4) {
    this.type = 1, this._$AH = D, this._$AN = void 0, this.element = t2, this.name = i3, this._$AM = e6, this.options = h4, s2.length > 2 || "" !== s2[0] || "" !== s2[1] ? (this._$AH = Array(s2.length - 1).fill(new String()), this.strings = s2) : this._$AH = D;
  }
  _$AI(t2, i3 = this, s2, e6) {
    const h4 = this.strings;
    let o4 = false;
    if (void 0 === h4) t2 = z(this, t2, i3, 0), o4 = !st(t2) || t2 !== this._$AH && t2 !== R, o4 && (this._$AH = t2);
    else {
      const e7 = t2;
      let n5, r5;
      for (t2 = h4[0], n5 = 0; n5 < h4.length - 1; n5++) r5 = z(this, e7[s2 + n5], i3, n5), r5 === R && (r5 = this._$AH[n5]), o4 ||= !st(r5) || r5 !== this._$AH[n5], r5 === D ? t2 = D : t2 !== D && (t2 += (r5 ?? "") + h4[n5 + 1]), this._$AH[n5] = r5;
    }
    o4 && !e6 && this.j(t2);
  }
  j(t2) {
    t2 === D ? this.element.removeAttribute(this.name) : this.element.setAttribute(this.name, t2 ?? "");
  }
};
var Y = class extends G {
  constructor() {
    super(...arguments), this.type = 3;
  }
  j(t2) {
    this.element[this.name] = t2 === D ? void 0 : t2;
  }
};
var Z = class extends G {
  constructor() {
    super(...arguments), this.type = 4;
  }
  j(t2) {
    this.element.toggleAttribute(this.name, !!t2 && t2 !== D);
  }
};
var q = class extends G {
  constructor(t2, i3, s2, e6, h4) {
    super(t2, i3, s2, e6, h4), this.type = 5;
  }
  _$AI(t2, i3 = this) {
    if ((t2 = z(this, t2, i3, 0) ?? D) === R) return;
    const s2 = this._$AH, e6 = t2 === D && s2 !== D || t2.capture !== s2.capture || t2.once !== s2.once || t2.passive !== s2.passive, h4 = t2 !== D && (s2 === D || e6);
    e6 && this.element.removeEventListener(this.name, this, s2), h4 && this.element.addEventListener(this.name, this, t2), this._$AH = t2;
  }
  handleEvent(t2) {
    "function" == typeof this._$AH ? this._$AH.call(this.options?.host ?? this.element, t2) : this._$AH.handleEvent(t2);
  }
};
var K = class {
  constructor(t2, i3, s2) {
    this.element = t2, this.type = 6, this._$AN = void 0, this._$AM = i3, this.options = s2;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  _$AI(t2) {
    z(this, t2);
  }
};
var Re = n3.litHtmlPolyfillSupport;
Re?.(B, et), (n3.litHtmlVersions ??= []).push("3.2.0");
var Q = (t2, i3, s2) => {
  const e6 = s2?.renderBefore ?? i3;
  let h4 = e6._$litPart$;
  if (void 0 === h4) {
    const t3 = s2?.renderBefore ?? null;
    e6._$litPart$ = h4 = new et(i3.insertBefore(lt(), t3), t3, void 0, s2 ?? {});
  }
  return h4._$AI(t2), h4;
};

// node_modules/lit-element/lit-element.js
var h3 = class extends b {
  constructor() {
    super(...arguments), this.renderOptions = { host: this }, this.o = void 0;
  }
  createRenderRoot() {
    const t2 = super.createRenderRoot();
    return this.renderOptions.renderBefore ??= t2.firstChild, t2;
  }
  update(t2) {
    const e6 = this.render();
    this.hasUpdated || (this.renderOptions.isConnected = this.isConnected), super.update(t2), this.o = Q(e6, this.renderRoot, this.renderOptions);
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
var r3 = (t2 = o3, e6, r5) => {
  const { kind: n5, metadata: i3 } = r5;
  let s2 = globalThis.litPropertyMetadata.get(i3);
  if (void 0 === s2 && globalThis.litPropertyMetadata.set(i3, s2 = /* @__PURE__ */ new Map()), s2.set(r5.name, t2), "accessor" === n5) {
    const { name: o4 } = r5;
    return { set(r6) {
      const n6 = e6.get.call(this);
      e6.set.call(this, r6), this.requestUpdate(o4, n6, t2);
    }, init(e7) {
      return void 0 !== e7 && this.P(o4, void 0, t2), e7;
    } };
  }
  if ("setter" === n5) {
    const { name: o4 } = r5;
    return function(r6) {
      const n6 = this[o4];
      e6.call(this, r6), this.requestUpdate(o4, n6, t2);
    };
  }
  throw Error("Unsupported decorator location: " + n5);
};
function n4(t2) {
  return (e6, o4) => "object" == typeof o4 ? r3(t2, e6, o4) : ((t3, e7, o5) => {
    const r5 = e7.hasOwnProperty(o5);
    return e7.constructor.createProperty(o5, r5 ? { ...t3, wrapped: true } : t3), r5 ? Object.getOwnPropertyDescriptor(e7, o5) : void 0;
  })(t2, e6, o4);
}

// node_modules/@lit/reactive-element/decorators/base.js
var e3 = (e6, t2, c4) => (c4.configurable = true, c4.enumerable = true, Reflect.decorate && "object" != typeof t2 && Object.defineProperty(e6, t2, c4), c4);

// node_modules/@lit/reactive-element/decorators/query.js
function e4(e6, r5) {
  return (n5, s2, i3) => {
    const o4 = (t2) => t2.renderRoot?.querySelector(e6) ?? null;
    if (r5) {
      const { get: e7, set: r6 } = "object" == typeof s2 ? n5 : i3 ?? (() => {
        const t2 = Symbol();
        return { get() {
          return this[t2];
        }, set(e8) {
          this[t2] = e8;
        } };
      })();
      return e3(n5, s2, { get() {
        let t2 = e7.call(this);
        return void 0 === t2 && (t2 = o4(this), (null !== t2 || this.hasUpdated) && r6.call(this, t2)), t2;
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

// js/utils.ts
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
      ...Array.from({ length: 7 }, (_2, i3) => ({
        label: `${i3 + 3}`,
        value: `${i3 + 3}`
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
export {
  RasterLayerEditor
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
*/
